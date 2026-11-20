function lp01_fidelity(reference, output)
    size(reference) == size(output) || throw(DimensionMismatch("Field grids must match."))
    a = @view reference[:, 1, :]
    b = @view output[:, 1, :]
    na, nb = sum(abs2, a), sum(abs2, b)
    isfinite(na) && isfinite(nb) && na > 0 && nb > 0 ||
        throw(ArgumentError("LP01 fields must have finite, positive norms."))
    f = abs2(dot(vec(a), vec(b))) / (na * nb)
    isfinite(f) && -1e-12 <= f <= 1+1e-12 || error("Invalid normalized overlap.")
    return clamp(f, 0.0, 1.0)
end

function dio_at(power_W; dz=search_dz)
    isfinite(power_W) && power_W > 0 || throw(ArgumentError("Power must be positive and finite."))
    key = (Float64(power_W), Float64(dz))
    haskey(evaluation_cache, key) && return evaluation_cache[key]
    cpu_start = ccall(:clock, Int64, ()) / 1e6
    measured = @timed begin
        input = gaussian_pulse(domain, settings["num_modes"];
            peak_power=power_W, fwhm=settings["pulse_fwhm_ps"],
            coefficients=launch_coefficients)
        nonlinear_run = @timed solve_mmgnlse(input, parameters_cp, dz)
        nonlinear_field = get_temporal_field(nonlinear_run.value; z=:final)
        F_IO = lp01_fidelity(input, nonlinear_field)
        D_IO = 1-F_IO
        photons = value(lp01_photons, nonlinear_run.value)
        input_photons = value(lp01_photons, nonlinear_run.value; z=:initial)
        @assert isfinite(input_photons) && input_photons > 0
        photon_retention = photons / input_photons
        total_photons = value(all_photons, nonlinear_run.value)
        modal_energy = vec(sum(abs2.(nonlinear_field); dims=(1, 3)))
        fractions = modal_energy ./ sum(modal_energy)
        @assert isfinite(photons) && photons > 0 && isfinite(total_photons)
        field_cache[key] = (; input=copy(input), nonlinear=copy(nonlinear_field))
        (; power_W=Float64(power_W), dz=Float64(dz), D_IO, F_IO,
            photons, input_photons, photon_retention, total_photons,
            lp01_photon_fraction=photons/total_photons,
            lp01_energy_fraction=fractions[1],
            mode2_energy_fraction=fractions[2], mode3_energy_fraction=fractions[3],
            output_input_energy_ratio=sum(modal_energy)/sum(abs2, input),
            nonlinear_seconds=nonlinear_run.time)
    end
    result = merge(measured.value, (; seconds=measured.time,
        cpu_seconds=ccall(:clock, Int64, ())/1e6-cpu_start,
        allocated_bytes=measured.bytes))
    evaluation_cache[key] = result
    push!(evaluations, result)
    save_rows("evaluations.csv", evaluations)
    @printf("P=%8.3f kW | DIO=%.8f | FIO=%.8f | N01=%.4e | dz=%.3g m | %.1f s\n",
        power_W/1000, result.D_IO, result.F_IO, result.photons, dz, result.seconds)
    flush(stdout)
    return result
end

function power_gradient(u; h=fd_step, dz=search_dz)
    lower, upper = power_bounds_W ./ reference_power_W
    plus_u, minus_u = clamp(u+h, lower, upper), clamp(u-h, lower, upper)
    plus = dio_at(reference_power_W*plus_u; dz)
    minus = dio_at(reference_power_W*minus_u; dz)
    return (plus.D_IO-minus.D_IO)/(plus_u-minus_u)
end

function diagnostic_fields(row)
    key = (row.power_W, row.dz)
    saved = field_cache[key]
    if !haskey(saved, :linear)
        linear_solution = solve_mmgnlse(saved.input, parameters_linear, row.dz)
        saved = merge(saved, (; linear=copy(get_temporal_field(linear_solution; z=:final))))
        field_cache[key] = saved
    end
    return saved
end

function optimize_power(; start_power_W=reference_power_W)
    lower, upper = power_bounds_W ./ reference_power_W
    u = clamp(start_power_W/reference_power_W, lower, upper)
    current = dio_at(u*reference_power_W)
    @assert current.photon_retention >= minimum_brightness_fraction
    history = [(iteration=0, power_W=current.power_W,
        D_IO=current.D_IO, F_IO=current.F_IO,
        photons=current.photons, input_photons=current.input_photons,
        photon_retention=current.photon_retention, gradient_used=NaN, step_u=0.0)]
    save_rows("optimization_history.csv", history)
    evaluation_start = length(evaluations)
    status = "iteration budget reached"
    started = time_ns()
    for iteration in 1:max_iterations
        if length(evaluations)-evaluation_start+3 > max_optimizer_evaluations
            status = "objective evaluation budget reached"
            break
        end
        g = power_gradient(u)
        projected_gradient = u-clamp(u-g, lower, upper)
        if abs(projected_gradient) < gradient_tolerance
            status = "projected gradient tolerance reached"
            break
        end
        step = clamp(-learning_rate*g,
            -max_power_change_W/reference_power_W, max_power_change_W/reference_power_W)
        accepted = false
        for attempt in 1:12
            length(evaluations)-evaluation_start >= max_optimizer_evaluations && break
            trial_u = clamp(u+step, lower, upper)
            actual_step = trial_u-u
            abs(actual_step) < 1e-10 && break
            trial = dio_at(trial_u*reference_power_W)
            if trial.photon_retention >= minimum_brightness_fraction &&
                    trial.D_IO <= current.D_IO + 1e-4*g*actual_step
                u, current = trial_u, trial
                push!(history, (; iteration, power_W=current.power_W,
                    D_IO=current.D_IO, F_IO=current.F_IO,
                    photons=current.photons, input_photons=current.input_photons,
                    photon_retention=current.photon_retention, gradient_used=g, step_u=actual_step))
                save_rows("optimization_history.csv", history)
                @printf("ACCEPT %2d: %.3f kW, DIO=%.8f\n", iteration, current.power_W/1000, current.D_IO)
                flush(stdout)
                accepted = true
                break
            end
            step /= 2
        end
        if !accepted
            status = "no acceptable step within line-search/evaluation budget"
            break
        end
    end
    return (; best=current, history, status,
        seconds=(time_ns()-started)/1e9,
        new_objective_evaluations=length(evaluations)-evaluation_start)
end

same_power(a,b) = isapprox(a.power_W,b.power_W;atol=1e-6,rtol=0)

function draw_pulse_panel(ax, x, curves, xlabel, ylabel)
    labels = ("Input", "Linear output (diagnostic)", "Nonlinear output")
    colors = ("#777777", "#cf7c32", "#366b9e")
    styles = (":", "--", "-")
    for j in eachindex(curves)
        ax.plot(x, curves[j]; label=labels[j], color=colors[j], linestyle=styles[j])
    end
    envelope = reduce((a, b) -> max.(a, b), [y./maximum(y) for y in curves])
    support = findall(envelope .> 1e-5)
    left, right = x[first(support)], x[last(support)]
    padding = max(0.1*(right-left), abs(x[2]-x[1]))
    ax.set_xlim(left-padding, right+padding)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(; fontsize=8)
    ax.grid(; alpha=0.2)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end

function plot_lp01_output(label, row)
    saved = diagnostic_fields(row)
    fields = (saved.input, saved.linear, saved.nonlinear)
    spectra = [fftshift(ifft(a, 1), 1) for a in fields]
    bin_power = [abs2.(a[:, 1, 1]) for a in spectra]
    temporal_power = [abs2.(a[:, 1, 1])./1000 for a in fields]
    npzwrite(joinpath(run_dir, label*"_fields.npz"), Dict(
        "time_ps" => times_ps, "detuning_THz" => detuning_THz,
        "wavelength_nm" => wavelengths_nm,
        "input" => saved.input, "linear" => saved.linear, "nonlinear" => saved.nonlinear,
        "power_W" => row.power_W, "dz_m" => row.dz))
    title = @sprintf("%s: %.3f kW, DIO=%.6f, dz=%.0f mm",
        label, row.power_W/1000, row.D_IO, row.dz*1000)

    fig, ax = subplots(; figsize=(6.8, 3.8), constrained_layout=true)
    draw_pulse_panel(ax, times_ps, temporal_power, "Time (ps)", "LP01 power (kW)")
    ax.set_title(title)
    savefig(joinpath(run_dir, label*"_time.png"); dpi=180)
    savefig(joinpath(run_dir, label*"_time.pdf"))
    display(fig)
    close(fig)

    fig, ax = subplots(; figsize=(6.8, 3.8), constrained_layout=true)
    draw_pulse_panel(ax, wavelengths_nm[wavelength_indices],
        [p[wavelength_indices] for p in bin_power],
        "Wavelength (nm)", "Power per frequency bin (W)")
    ax.set_title(title)
    savefig(joinpath(run_dir, label*"_spectrum.png"); dpi=180)
    savefig(joinpath(run_dir, label*"_spectrum.pdf"))
    display(fig)
    close(fig)
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

function plot_beam_preview(label, row)
    saved = diagnostic_fields(row)
    maps = [peak_beam_map(a) for a in (saved.input, saved.linear, saved.nonlinear)]
    labels = ("Input", "Linear output (diagnostic)", "Nonlinear output")
    observed_max = maximum(maximum(m.intensity) for m in maps)
    displayed_intensity_maxima[label*"_beam"] = observed_max
    common_max = beam_intensity_limits_W_per_um2[2]
    @assert observed_max <= common_max "Increase the common limit in plot_settings.toml."
    fig, axes = subplots(1, 3; figsize=(18, 6), constrained_layout=true)
    meshes = []
    for ii = 1:3
        ax = axes[ii]
        mesh = ax.pcolormesh(x.*1e6, y.*1e6, maps[ii].intensity;
            cmap=beam_colormap, vmin=beam_intensity_limits_W_per_um2[1], vmax=common_max, shading="auto", rasterized=true)
        push!(meshes, mesh)
        ax.set_title(@sprintf("%s\npeak %.2f kW at t = %.3f ps",
            labels[ii], maps[ii].peak_power_W/1000, maps[ii].peak_time_ps))
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.set_aspect("equal")
        theta = range(0, 2pi; length=181)
        ax.plot(settings["core_radius_m"].*1e6.*cos.(theta),
            settings["core_radius_m"].*1e6.*sin.(theta);
            color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    end
    fig.colorbar(meshes[end]; ax=collect(axes), shrink=0.78, ticks=beam_color_ticks,
        label="Intensity at each pulse's peak (W/μm²)")
    fig.suptitle(@sprintf("%s — %.3f kW input; output dz = %.0f mm",
        label, row.power_W/1000, row.dz*1000))
    savefig(joinpath(run_dir, label*"_beam_preview.png"); dpi=160)
    savefig(joinpath(run_dir, label*"_beam_preview.pdf"))
    display(fig)
    close(fig)
    npzwrite(joinpath(run_dir, label*"_beam_maps.npz"), Dict(
        "x_um"=>x.*1e6, "y_um"=>y.*1e6,
        "input_W_per_um2"=>maps[1].intensity,
        "linear_W_per_um2"=>maps[2].intensity,
        "nonlinear_W_per_um2"=>maps[3].intensity,
        "peak_times_ps"=>[m.peak_time_ps for m in maps],
        "peak_powers_W"=>[m.peak_power_W for m in maps],
        "shared_color_min_W_per_um2"=>beam_intensity_limits_W_per_um2[1],
        "shared_color_max_W_per_um2"=>common_max))
    @printf("%s: spatial integrals match the three pulse peak powers.\n", label)
end

function plot_individual_mode_intensity(label, row)
    saved = diagnostic_fields(row)
    fields = (saved.input, saved.linear, saved.nonlinear)
    field_labels = ("Input", "Linear output", "Nonlinear output")
    peak_indices = [argmax(vec(sum(abs2.(a); dims=(2, 3)))) for a in fields]
    maps = [abs2.(modes[:, :, mm]) .* sum(abs2, a[peak_indices[jj], mm, :]) ./ 1e12
        for (jj, a) in enumerate(fields), mm in 1:3]
    observed_max = maximum(maximum(a) for a in maps)
    displayed_intensity_maxima[label*"_individual_modes"] = observed_max
    @assert observed_max <= beam_intensity_limits_W_per_um2[2]
    fig, axes = subplots(3, 3; figsize=(12, 10), constrained_layout=true)
    mesh = nothing
    theta = range(0, 2pi; length=181)
    for jj in 1:3, mm in 1:3
        ax = axes[jj, mm]
        mesh = ax.pcolormesh(x.*1e6, y.*1e6, maps[jj, mm]; cmap=beam_colormap,
            vmin=beam_intensity_limits_W_per_um2[1], vmax=beam_intensity_limits_W_per_um2[2],
            shading="auto", rasterized=true)
        ax.set_title("$(field_labels[jj]) — mode $mm")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.set_aspect("equal")
        ax.plot(settings["core_radius_m"].*1e6.*cos.(theta),
            settings["core_radius_m"].*1e6.*sin.(theta);
            color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    end
    fig.colorbar(mesh; ax=vec(collect(axes)), shrink=0.78, ticks=beam_color_ticks,
        label="Individual-mode intensity (W/μm²)")
    fig.suptitle(@sprintf("%s — individual modes at each field's peak total-power time", label))
    savefig(joinpath(run_dir, label*"_individual_modes.png"); dpi=160)
    savefig(joinpath(run_dir, label*"_individual_modes.pdf"))
    display(fig)
    close(fig)
end
