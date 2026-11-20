power_label(P) = replace(@sprintf("%.3fkw", P/1000), "."=>"p")

case_dir(P) = joinpath(run_dir, power_label(P))

function write_toml(path, data)
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function csv_cell(value)
    s = string(value)
    return occursin(r"[,\"\n\r]", s) ? "\""*replace(s, "\""=>"\"\"")*"\"" : s
end

function save_rows(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ","))
        for row in rows
            println(io, join(csv_cell.(collect(values(row))), ","))
        end
    end
end

function simulation_context(nt::Int, representation::Symbol=:cp)
    get!(context_cache, (nt, representation)) do
        representation == :cp || error("This run uses the unchanged CP representation.")
        overlap = S_cp
        grid = TimeGrid(nt, settings["time_window_ps"])
        domain = MMGNLSEDomain(degrees_of_freedom(:time, :space), grid)
        nonlinear = MMGNLSEParameters(domain; length=settings["fiber_length_m"],
            beta=properties.beta, S=overlap, n2=properties.n2,
            omega0=properties.omega0, raman=properties.raman)
        linear = MMGNLSEParameters(domain; length=settings["fiber_length_m"],
            beta=properties.beta, S=overlap, n2=0.0,
            omega0=properties.omega0, raman=properties.raman)
        photons = SpectralPhotonNumber(filter=ones(nt), modes=[1])
        (; grid, domain, nonlinear, linear, photons)
    end
end

function canonical_coefficients(c)
    length(c) == 3 || throw(ArgumentError("Supply three coefficients."))
    v = Float64.(collect(c))
    all(isfinite, v) && all(x -> 0 <= x <= 100, v) ||
        throw(ArgumentError("Each raw coefficient must be in [0,100]."))
    sum(v) > 0 || throw(ArgumentError("The all-zero launch has no defined mixture."))
    return 100 .* v ./ sum(v)
end

function project_simplex(v)
    length(v) == 3 && all(isfinite, v) || throw(ArgumentError("Invalid update vector."))
    ordered = sort(Float64.(v); rev=true)
    partial = cumsum(ordered)
    rho = findlast(j -> ordered[j] > (partial[j]-1)/j, 1:3)
    theta = (partial[rho]-1)/rho
    w = max.(Float64.(v) .- theta, 0.0)
    return w ./ sum(w)
end

projected_gradient(u, g) = u .- project_simplex(u .- g)

coefficients(row) = [row.coefficient_1, row.coefficient_2, row.coefficient_3]

coeff_label(c) = @sprintf("[%.3f, %.3f, %.3f]", c[1], c[2], c[3])

function field_fidelity(reference, output; selected_modes=1:1, allow_empty=false)
    size(reference) == size(output) || throw(DimensionMismatch("Field grids differ."))
    a, b = @view(reference[:,selected_modes,:]), @view(output[:,selected_modes,:])
    na, nb = sum(abs2,a), sum(abs2,b)
    if !(isfinite(na) && isfinite(nb) && na > 0 && nb > 0)
        allow_empty && return NaN
        error("Selected fields require finite, positive norms.")
    end
    F = abs2(dot(vec(a),vec(b)))/(na*nb)
    isfinite(F) && -1e-12 <= F <= 1+1e-12 || error("Invalid field fidelity.")
    return clamp(F,0.0,1.0)
end

budget(P, phase) = phase == :optimization ? max_optimizer_evaluations : max_check_evaluations

remaining(P, phase) = budget(P,phase)-used[P][phase]

function evaluation_key(c, P, dz, nt, representation)
    v = canonical_coefficients(c)
    return (Float64(P), Tuple(round.(v;digits=12)), Float64(dz), Int(nt), representation)
end

function dio_at(c, P; dz=search_dz, nt=settings["Nt"], representation=:cp,
                phase=:validation)
    P in power_cases_W || throw(ArgumentError("Use one of the declared fixed powers."))
    phase in (:optimization,:validation) || error("Unknown budget bucket.")
    v = canonical_coefficients(c)
    key = evaluation_key(v,P,dz,nt,representation)
    haskey(evaluation_cache,key) && return evaluation_cache[key]
    remaining(P,phase) > 0 || error("$(phase) evaluation budget reached at $(P/1000) kW")
    used[P][phase] += 1
    attempt_id[] += 1
    id = attempt_id[]
    try
        context = simulation_context(Int(nt),representation)
        measured = @timed begin
            input = gaussian_pulse(context.domain,3; peak_power=P,
                fwhm=settings["pulse_fwhm_ps"],coefficients=ComplexF64.(v))
            nonlinear_run = @timed solve_mmgnlse(input,context.nonlinear,dz)
            nonlinear = get_temporal_field(nonlinear_run.value;z=:final)
            all(isfinite,nonlinear) || error("Nonfinite output field.")
            F_IO = field_fidelity(input,nonlinear; selected_modes=1:1, allow_empty=true)
            objective_defined = isfinite(F_IO)
            photons = value(context.photons,nonlinear_run.value)
            photons >= 0 && isfinite(photons) || error("Invalid photon number.")
            input_energy = vec(sum(abs2.(input);dims=(1,3)))
            output_energy = vec(sum(abs2.(nonlinear);dims=(1,3)))
            pin, pout = input_energy./sum(input_energy), output_energy./sum(output_energy)
            @assert isapprox(maximum(sum(abs2.(input);dims=(2,3))),P;rtol=1e-12)
            @assert isapprox(pin, v.^2 ./ sum(abs2,v);rtol=1e-12,atol=1e-14)
            temporal_power = vec(sum(abs2.(nonlinear);dims=(2,3)))
            spectrum = fftshift(ifft(nonlinear,1),1)
            spectral_power = vec(sum(abs2.(spectrum);dims=(2,3)))
            t, omega = collect(time_axis(context.grid)),collect(frequency_axis(context.grid))
            time_edge = sum(temporal_power[abs.(t) .> 0.45*settings["time_window_ps"]])/sum(temporal_power)
            frequency_edge = sum(spectral_power[abs.(omega) .> 0.8*maximum(abs,omega)])/sum(spectral_power)
            field_cache[id] = (;input=copy(input),nonlinear=copy(nonlinear))
            (; id, coefficient_1=v[1],coefficient_2=v[2],coefficient_3=v[3],
                power_W=Float64(P),dz=Float64(dz),nt=Int(nt),representation=string(representation),
                phase=string(phase),D_IO=objective_defined ? 1-F_IO : 1.0,F_IO,objective_defined,
                photons,
                input_mode1_fraction=pin[1],input_mode2_fraction=pin[2],input_mode3_fraction=pin[3],
                mode1_energy_fraction=pout[1],mode2_energy_fraction=pout[2],mode3_energy_fraction=pout[3],
                output_input_energy_ratio=sum(output_energy)/sum(input_energy),
                time_edge_fraction=time_edge,frequency_edge_fraction=frequency_edge,
                nonlinear_seconds=nonlinear_run.time)
        end
        result = merge(measured.value,(;seconds=measured.time,allocated_bytes=measured.bytes))
        evaluation_cache[key] = result
        push!(evaluations,result)
        save_rows(joinpath(run_dir,"evaluations.csv"),evaluations)
        save_rows(joinpath(case_dir(P),"evaluations.csv"),filter(r->r.power_W==P,evaluations))
        @printf("%7.3f kW %s D_IO=%.8f | %s %.1f mm Nt=%d | %.2f s\n",
            P/1000,coeff_label(v),result.D_IO,string(representation),1000dz,nt,result.seconds)
        flush(stdout)
        return result
    catch err
        push!(failures,(;id,power_W=P,coefficient_1=v[1],coefficient_2=v[2],coefficient_3=v[3],
            dz,nt,representation=string(representation),phase=string(phase),error=sprint(showerror,err)))
        save_rows(joinpath(run_dir,"failed_evaluations.csv"),failures)
        rethrow()
    end
end

function attach_linear_diagnostic!(row)
    saved=field_cache[row.id]
    hasproperty(saved,:linear) && return saved
    context=simulation_context(row.nt)
    measured=@timed solve_mmgnlse(saved.input,context.linear,row.dz)
    linear=get_temporal_field(measured.value;z=:final)
    all(isfinite,linear) || error("Nonfinite linear diagnostic field.")
    saved=merge(saved,(;linear=copy(linear)))
    field_cache[row.id]=saved
    push!(linear_diagnostics,(;evaluation_id=row.id,power_W=row.power_W,dz_m=row.dz,
        F_linear_output_diagnostic=field_fidelity(linear,saved.nonlinear),
        linear_seconds=measured.time))
    save_rows(joinpath(run_dir,"linear_diagnostics.csv"),linear_diagnostics)
    return saved
end

function coefficient_stencil(c;h=fd_step)
    lo,hi=coefficient_bounds
    lo <= c <= hi && 0 < h <= (hi-lo)/2 || error("Invalid finite-difference stencil.")
    if c-h >= lo && c+h <= hi
        return ([c-h,c+h],[-1.0,1.0]./(2h))
    elseif c-h < lo
        return ([c,c+h,c+2h],[-3.0,4.0,-1.0]./(2h))
    else
        return ([c,c-h,c-2h],[3.0,-4.0,1.0]./(2h))
    end
end

function gradient_probes(u;h=fd_step)
    c=100 .* project_simplex(u)
    probes=Tuple{Int,Vector{Float64},Float64}[]
    for j in 1:3
        points,weights=coefficient_stencil(c[j];h)
        for (p,w) in zip(points,weights)
            v=copy(c);v[j]=p
            push!(probes,(j,v,100w))
        end
    end
    probes
end

function new_probe_count(u,P;h=fd_step,dz=search_dz)
    keys_to_try=unique(evaluation_key(v,P,dz,settings["Nt"],:cp) for (_,v,_) in gradient_probes(u;h))
    count(k->!haskey(evaluation_cache,k),keys_to_try)
end

function launch_gradient(u,P;h=fd_step,dz=search_dz,phase=:validation)
    new_probe_count(u,P;h,dz) <= remaining(P,phase) || error("Insufficient budget for a complete gradient.")
    g=zeros(3)
    for (j,v,w) in gradient_probes(u;h)
        g[j] += w*dio_at(v,P;dz,phase).D_IO
    end
    g
end

function history_row(iteration,row;g=fill(NaN,3),pg=NaN,step=0.0,alpha=0.0)
    c=coefficients(row)
    (;iteration,evaluation_id=row.id,power_W=row.power_W,
        coefficient_1=c[1],coefficient_2=c[2],coefficient_3=c[3],D_IO=row.D_IO,
        input_mode1_fraction=row.input_mode1_fraction,
        input_mode2_fraction=row.input_mode2_fraction,
        input_mode3_fraction=row.input_mode3_fraction,
        brightness_ratio=row.photons/preflights[row.power_W].fine.photons,
        previous_gradient_u1=g[1],previous_gradient_u2=g[2],previous_gradient_u3=g[3],
        previous_projected_gradient_norm=pg,max_coefficient_step=step,
        backtracking_multiplier=alpha,optimization_evaluations=used[row.power_W][:optimization])
end

function optimize_launch(P)
    current=preflights[P].baseline
    u=coefficients(current)./100
    history=[history_row(0,current)]
    trials=NamedTuple[]
    status="iteration_budget"
    for iteration in 1:max_iterations
        # Reserve enough budget for a complete gradient and one candidate.
        if remaining(P,:optimization) < new_probe_count(u,P)+1
            status="evaluation_budget";break
        end
        g=launch_gradient(u,P;phase=:optimization)
        pg=norm(projected_gradient(u,g),Inf)
        if pg<=gradient_tolerance
            status="small_projected_gradient_search_grid";break
        end
        direction=project_simplex(u.-learning_rate.*g).-u
        largest=100norm(direction,Inf)
        largest>0 || (status="zero_projected_step";break)
        direction .*= min(1.0,max_coefficient_change/largest)
        accepted=false
        for attempt in 0:settings["max_backtracking_halvings"]
            alpha=0.5^attempt
            trial_u=project_simplex(u.+alpha.*direction)
            delta=trial_u.-u
            c=100 .* trial_u
            key=evaluation_key(c,P,search_dz,settings["Nt"],:cp)
            if !haskey(evaluation_cache,key) && remaining(P,:optimization)==0
                status="evaluation_budget";break
            end
            trial=dio_at(c,P;phase=:optimization)
            step=100norm(delta,Inf)
            feasible=trial.objective_defined && trial.photons>=brightness_floors[P]
            slope=dot(g,delta)
            armijo=slope<0 && trial.D_IO<=current.D_IO+settings["armijo_c1"]*slope
            accepted=feasible && armijo && trial.D_IO<current.D_IO-minimum_decrease
            @assert step<=max_coefficient_change+1e-10
            push!(trials,(;iteration,attempt,evaluation_id=trial.id,
                coefficient_1=c[1],coefficient_2=c[2],coefficient_3=c[3],
                previous_D_IO=current.D_IO,trial_D_IO=trial.D_IO,
                gradient_dot_step=slope,alpha,max_coefficient_step=step,
                feasible,armijo,accepted))
            save_rows(joinpath(case_dir(P),"line_search.csv"),trials)
            if accepted
                current=trial;u=coefficients(current)./100
                push!(history,history_row(iteration,current;g,pg,step,alpha))
                save_rows(joinpath(case_dir(P),"history.csv"),history)
                @printf("ACCEPT %.3f kW iteration %d: %s, D_IO %.8f, max move %.4f\n",
                    P/1000,iteration,coeff_label(coefficients(current)),current.D_IO,step)
                flush(stdout)
                break
            end
        end
        if !accepted
            status=="evaluation_budget" || (status="line_search_failed")
            break
        end
    end
    save_rows(joinpath(case_dir(P),"history.csv"),history)
    write_toml(joinpath(case_dir(P),"optimization.toml"),Dict(
        "power_W"=>P,"stop_reason"=>status,"accepted_steps"=>length(history)-1,
        "selected_coefficients"=>coefficients(current),"selected_D_IO_search"=>current.D_IO,
        "new_optimization_evaluations"=>used[P][:optimization]))
    (;best=current,history,trials,status)
end

function comparison_row(label,row,baseline)
    c=coefficients(row)
    (;power_W=row.power_W,launch=label,coefficient_1=c[1],coefficient_2=c[2],
        coefficient_3=c[3],D_IO=row.D_IO,F_IO=row.F_IO,
        brightness_ratio=row.photons/baseline.photons,dz_m=row.dz,nt=row.nt,
        representation=row.representation,evaluation_id=row.id)
end

function finish_figure(fig,path;dpi=150)
    fig.savefig(path*".png";dpi)
    fig.savefig(path*".pdf")
    display(fig)
    close(fig)
end

function tidy_axis(ax)
    ax.grid(;alpha=0.18)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end

triangle_xy(c)=let p=c.^2 ./ sum(abs2,c);(p[2]+p[3]/2,sqrt(3)*p[3]/2);end

function draw_pulse_panel(ax,x,curves,xlabel,ylabel)
    for j in eachindex(curves)
        ax.plot(x,curves[j];label=curve_labels[j],color=curve_colors[j],linestyle=curve_styles[j])
    end
    max_value=maximum(maximum(y) for y in curves)
    if max_value>1e-20
        envelope=reduce((a,b)->max.(a,b),curves)./max_value
        support=findall(envelope.>1e-5)
        left,right=x[first(support)],x[last(support)]
        padding=max(0.1*(right-left),abs(x[2]-x[1]))
        ax.set_xlim(left-padding,right+padding)
    end
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.legend(;fontsize=7.5);tidy_axis(ax)
end

function plot_modal_output(label,row)
    saved=field_cache[row.id]
    fields=(saved.input,saved.linear,saved.nonlinear)
    spectra=[fftshift(ifft(a,1),1) for a in fields]
    npzwrite(joinpath(case_dir(row.power_W),label*"_fields.npz"),Dict(
        "time_ps"=>times_ps,"wavelength_nm"=>wavelengths_nm,
        "input"=>saved.input,"linear"=>saved.linear,"nonlinear"=>saved.nonlinear,
        "power_W"=>row.power_W,"dz_m"=>row.dz,"launch_coefficients"=>coefficients(row)))
    caption=@sprintf("%s | %.3f kW | %s | LP01 D_IO %.6f | 1 mm",
        label,row.power_W/1000,coeff_label(coefficients(row)),row.D_IO)
    for kind in ("time","spectrum")
        fig,axes=subplots(1,3;figsize=(16,4.3),constrained_layout=true)
        for m in 1:3
            if kind=="time"
                curves=[vec(sum(abs2.(a[:,m,:]);dims=2))./1000 for a in fields]
                draw_pulse_panel(axes[m],times_ps,curves,"Time (ps)","Modal power (kW)")
            else
                curves=[vec(sum(abs2.(a[:,m,:]);dims=2))[wavelength_indices] for a in spectra]
                draw_pulse_panel(axes[m],wavelengths_nm[wavelength_indices],curves,
                    "Wavelength (nm)","Power per frequency bin (W)")
            end
            axes[m].set_title("Mode $m: "*mode_names[m])
        end
        fig.suptitle(caption;fontsize=11)
        finish_figure(fig,joinpath(case_dir(row.power_W),label*"_"*kind))
    end
end

function peak_beam_map(modal_field)
    power_t=vec(sum(abs2.(modal_field);dims=(2,3)));k=argmax(power_t)
    flat_modes=reshape(modes,:,size(modes,3))
    intensity=zeros(Float64,size(modes,1),size(modes,2))
    for polarization in Base.axes(modal_field,3)
        amplitudes=modal_field[k,:,polarization]
        spatial_field=reshape(flat_modes*amplitudes,size(intensity))
        intensity .+= abs2.(spatial_field)
    end
    intensity ./= 1e12 # W/m² to W/μm²
    reconstructed=sum(intensity)*(x[2]-x[1])*(y[2]-y[1])*1e12
    @assert isapprox(reconstructed,power_t[k];rtol=1e-6)
    (;intensity,peak_time_ps=times_ps[k],peak_power_W=power_t[k],reconstructed_power_W=reconstructed)
end
