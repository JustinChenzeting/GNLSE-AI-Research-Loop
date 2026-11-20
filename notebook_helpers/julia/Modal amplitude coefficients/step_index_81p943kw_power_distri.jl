launch_coefficients(c) = ComplexF64[c, (100-c)*cis(relative_phase_rad), 0]

function save_rows(filename, rows)
    isempty(rows) && return
    open(joinpath(run_dir, filename), "w") do io
        println(io, join(string.(keys(first(rows))), ","))
        for row in rows
            println(io, join(values(row), ","))
        end
    end
end

function field_fidelity(reference, output; selected_modes=1:2)
    size(reference) == size(output) || throw(DimensionMismatch("Field grids must match."))
    a = @view reference[:, selected_modes, :]
    b = @view output[:, selected_modes, :]
    na, nb = sum(abs2, a), sum(abs2, b)
    isfinite(na) && isfinite(nb) && na > 0 && nb > 0 ||
        throw(ArgumentError("Combined fields must have finite, positive norms."))
    f = abs2(dot(vec(a), vec(b))) / (na * nb)
    isfinite(f) && -1e-12 <= f <= 1+1e-12 || error("Invalid normalized overlap.")
    return clamp(f, 0.0, 1.0)
end

function coefficient_stencil(c; h=fd_step)
    lower, upper = coefficient_bounds
    lower <= c <= upper || throw(ArgumentError("Coefficient is outside its bounds."))
    0 < h <= (upper-lower)/2 || throw(ArgumentError("Invalid finite-difference step."))
    if c-h >= lower && c+h <= upper
        return ([c-h, c+h], [-1.0, 1.0]./(2h))
    elseif c-h < lower
        return ([c, c+h, c+2h], [-3.0, 4.0, -1.0]./(2h))
    else
        return ([c, c-h, c-2h], [3.0, -4.0, 1.0]./(2h))
    end
end

function dnl_at(c; dz=search_dz)
    lower, upper = coefficient_bounds
    isfinite(c) && lower <= c <= upper || throw(ArgumentError("Invalid coefficient."))
    key = (Float64(c), Float64(dz))
    haskey(evaluation_cache, key) && return evaluation_cache[key]
    cpu_start = ccall(:clock, Int64, ()) / 1e6
    measured = @timed begin
        input = gaussian_pulse(domain, settings["num_modes"];
            peak_power=fixed_power_W, fwhm=settings["pulse_fwhm_ps"],
            coefficients=launch_coefficients(c))
        # Propagate the SAME candidate through both branches.
        linear_run = @timed solve_mmgnlse(input, parameters_linear, dz)
        nonlinear_run = @timed solve_mmgnlse(input, parameters_cp, dz)
        linear_field = get_temporal_field(linear_run.value; z=:final)
        nonlinear_field = get_temporal_field(nonlinear_run.value; z=:final)
        F_NL = field_fidelity(linear_field, nonlinear_field)
        D_NL = 1-F_NL
        photons = value(detected_photons, nonlinear_run.value)
        total_photons = value(all_photons, nonlinear_run.value)
        input_energy = vec(sum(abs2.(input); dims=(1, 3)))
        output_energy = vec(sum(abs2.(nonlinear_field); dims=(1, 3)))
        input_fractions = input_energy ./ sum(input_energy)
        output_fractions = output_energy ./ sum(output_energy)
        @assert isapprox(maximum(sum(abs2.(input); dims=(2, 3))),
            fixed_power_W; rtol=1e-12)
        @assert input_fractions[3] == 0
        @assert isfinite(photons) && photons > 0 && isfinite(total_photons)
        field_cache[key] = (; input=copy(input), linear=copy(linear_field),
            nonlinear=copy(nonlinear_field))
        (; coefficient_1=Float64(c), coefficient_2=Float64(100-c), coefficient_3=0.0,
            power_W=fixed_power_W, dz=Float64(dz), D_NL, F_NL,
            D_NL_all_modes=1-field_fidelity(linear_field, nonlinear_field; selected_modes=1:3),
            F_input=field_fidelity(input, nonlinear_field),
            photons, total_photons, detected_photon_fraction=photons/total_photons,
            input_mode1_fraction=input_fractions[1], input_mode2_fraction=input_fractions[2],
            mode1_energy_fraction=output_fractions[1],
            mode2_energy_fraction=output_fractions[2], mode3_energy_fraction=output_fractions[3],
            output_input_energy_ratio=sum(output_energy)/sum(input_energy),
            linear_seconds=linear_run.time, nonlinear_seconds=nonlinear_run.time)
    end
    result = merge(measured.value, (; seconds=measured.time,
        cpu_seconds=ccall(:clock, Int64, ())/1e6-cpu_start,
        allocated_bytes=measured.bytes))
    evaluation_cache[key] = result
    push!(evaluations, result)
    save_rows("evaluations.csv", evaluations)
    @printf("c=[%7.3f, %7.3f, 0] | DNL=%.8f | N12=%.4e | dz=%.2f mm | %.1f s\n",
        c, 100-c, result.D_NL, result.photons, dz*1000, result.seconds)
    flush(stdout)
    return result
end

function launch_gradient(u; h=fd_step, dz=search_dz)
    points, weights = coefficient_stencil(100u; h)
    # These paired simulations include BOTH changed coefficients and both branches.
    return 100sum(w*dnl_at(c; dz).D_NL for (c, w) in zip(points, weights))
end

function history_row(iteration, current; gradient_used=NaN, step_u=0.0)
    return (; iteration, coefficient_1=current.coefficient_1,
        coefficient_2=current.coefficient_2, coefficient_3=0.0,
        D_NL=current.D_NL, F_NL=current.F_NL, photons=current.photons,
        gradient_used, step_u)
end

function optimize_launch(; start_coefficient=baseline_coefficient)
    u = clamp(start_coefficient/100, 0.0, 1.0)
    current = dnl_at(100u)
    @assert current.photons >= brightness_floor
    history = [history_row(0, current)]
    save_rows("optimization_history.csv", history)
    evaluation_start = length(evaluations)
    status = "iteration budget reached"
    started = time_ns()
    for iteration in 1:max_iterations
        points, _ = coefficient_stencil(100u)
        gradient_cost = count(c -> !haskey(evaluation_cache,
            (Float64(c), Float64(search_dz))), points)
        if length(evaluations)-evaluation_start+gradient_cost+1 > max_optimizer_evaluations
            status = "objective evaluation budget reached"
            break
        end
        g = launch_gradient(u)
        projected_gradient = u-clamp(u-g, 0.0, 1.0)
        if abs(projected_gradient) < gradient_tolerance
            status = "projected gradient tolerance reached"
            break
        end
        step = clamp(-learning_rate*g,
            -max_coefficient_change/100, max_coefficient_change/100)
        accepted = false
        budget_hit = false
        for attempt in 1:12
            if length(evaluations)-evaluation_start >= max_optimizer_evaluations
                budget_hit = true
                break
            end
            trial_u = clamp(u+step, 0.0, 1.0)
            actual_step = trial_u-u
            abs(actual_step) < 1e-10 && break
            trial = dnl_at(100trial_u)
            if trial.photons >= brightness_floor &&
                    trial.D_NL < current.D_NL &&
                    trial.D_NL <= current.D_NL + 1e-4*g*actual_step
                u, current = trial_u, trial
                push!(history, history_row(iteration, current;
                    gradient_used=g, step_u=actual_step))
                save_rows("optimization_history.csv", history)
                @printf("ACCEPT %2d: [%.5f, %.5f, 0], DNL=%.8f\n",
                    iteration, current.coefficient_1, current.coefficient_2, current.D_NL)
                flush(stdout)
                accepted = true
                break
            end
            step /= 2
        end
        if !accepted
            status = budget_hit ? "objective evaluation budget reached" : "line search failed"
            break
        end
    end
    return (; best=current, history, status, seconds=(time_ns()-started)/1e9,
        new_objective_evaluations=length(evaluations)-evaluation_start)
end

function draw_pulse_panel(ax, x, curves, xlabel, ylabel)
    for j in eachindex(curves)
        ax.plot(x, curves[j]; label=curve_labels[j],
            color=curve_colors[j], linestyle=curve_styles[j])
    end
    max_value = maximum(maximum(y) for y in curves)
    if max_value > 1e-20
        envelope = reduce((a, b) -> max.(a, b), curves)./max_value
        support = findall(envelope .> 1e-5)
        left, right = x[first(support)], x[last(support)]
        padding = max(0.1*(right-left), abs(x[2]-x[1]))
        ax.set_xlim(left-padding, right+padding)
    end
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(; fontsize=8)
    ax.grid(; alpha=0.2)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end

function plot_modal_output(label, row)
    saved = field_cache[(row.coefficient_1, row.dz)]
    fields = (saved.input, saved.linear, saved.nonlinear)
    spectra = [fftshift(ifft(a, 1), 1) for a in fields]
    npzwrite(joinpath(run_dir, label*"_fields.npz"), Dict(
        "time_ps"=>times_ps, "wavelength_nm"=>wavelengths_nm,
        "input"=>saved.input, "linear"=>saved.linear, "nonlinear"=>saved.nonlinear,
        "power_W"=>fixed_power_W, "dz_m"=>row.dz,
        "launch_coefficients"=>launch_coefficients(row.coefficient_1)))
    caption = @sprintf("%s: [%.3f, %.3f, 0], %.3f kW; DNL=%.6f; dz=1 mm",
        label, row.coefficient_1, row.coefficient_2, fixed_power_W/1000, row.D_NL)
    for kind in ("time", "spectrum")
        fig, axes = subplots(1, 3; figsize=(16.5, 4.2), constrained_layout=true)
        for m in 1:3
            if kind == "time"
                curves = [vec(sum(abs2.(a[:, m, :]); dims=2))./1000 for a in fields]
                draw_pulse_panel(axes[m], times_ps, curves, "Time (ps)", "Modal power (kW)")
            else
                curves = [vec(sum(abs2.(a[:, m, :]); dims=2))[wavelength_indices] for a in spectra]
                draw_pulse_panel(axes[m], wavelengths_nm[wavelength_indices],
                    curves, "Wavelength (nm)", "Power per frequency bin (W)")
            end
            axes[m].set_title("Mode $m: "*mode_names[m])
            if m == 3
                axes[m].text(0.03, 0.96, @sprintf("Output energy fraction: %.2e",
                    row.mode3_energy_fraction); transform=axes[m].transAxes,
                    va="top", fontsize=8)
            end
        end
        fig.suptitle(caption; fontsize=12)
        savefig(joinpath(run_dir, label*"_"*kind*".png"); dpi=170)
        savefig(joinpath(run_dir, label*"_"*kind*".pdf"))
        display(fig)
        close(fig)
    end
end

function peak_beam_map(modal_field)
    # Find the peak of total power, summed over modes and polarizations.
    power_t = vec(sum(abs2.(modal_field); dims=(2, 3)))
    peak_index = argmax(power_t)
    flat_modes = reshape(modes, :, size(modes, 3))
    intensity = zeros(Float64, size(modes, 1), size(modes, 2))
    for polarization in Base.axes(modal_field, 3)
        amplitudes = modal_field[peak_index, :, polarization]
        # Add complex modal fields BEFORE squaring: retain interference.
        spatial_field = reshape(flat_modes*amplitudes, size(intensity))
        intensity .+= abs2.(spatial_field)
    end
    intensity ./= 1e12  # W/m² -> W/μm²
    area_um2 = (x[2]-x[1])*(y[2]-y[1])*1e12
    reconstructed_power_W = sum(intensity)*area_um2
    @assert isapprox(reconstructed_power_W, power_t[peak_index]; rtol=1e-6)
    return (; intensity, peak_time_ps=times_ps[peak_index],
        peak_power_W=power_t[peak_index], reconstructed_power_W)
end
