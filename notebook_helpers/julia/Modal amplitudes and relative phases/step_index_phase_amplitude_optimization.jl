function simulation_context(nt::Int, representation::Symbol=:cp)
    get!(context_cache, (nt, representation)) do
        overlap = representation == :cp ? S_cp :
            representation == :dense ? properties.S : error("Unknown representation")
        grid = TimeGrid(nt, settings["time_window_ps"])
        domain = MMGNLSEDomain(degrees_of_freedom(:time, :space), grid)
        nonlinear = MMGNLSEParameters(domain; length=settings["fiber_length_m"],
            beta=properties.beta, S=overlap, n2=properties.n2,
            omega0=properties.omega0, raman=properties.raman)
        linear = MMGNLSEParameters(domain; length=settings["fiber_length_m"],
            beta=properties.beta, S=overlap, n2=0.0,
            omega0=properties.omega0, raman=properties.raman)
        photons = SpectralPhotonNumber(filter=ones(nt), modes=:all)
        photons12 = SpectralPhotonNumber(filter=ones(nt), modes=[1,2])
        (; grid, domain, nonlinear, linear, photons, photons12)
    end
end

function field_fidelity(reference, output; selected_modes=1:3, allow_empty=false)
    size(reference) == size(output) || throw(DimensionMismatch("Field grids differ."))
    a, b = @view(reference[:,selected_modes,:]), @view(output[:,selected_modes,:])
    na, nb = sum(abs2,a), sum(abs2,b)
    if !(isfinite(na) && isfinite(nb) && na > 0 && nb > 0)
        allow_empty && return NaN
        error("All-mode fields require finite, positive norms.")
    end
    F = abs2(dot(vec(a),vec(b)))/(na*nb)
    isfinite(F) && -1e-12 <= F <= 1+1e-12 || error("Invalid field fidelity.")
    return clamp(F,0.0,1.0)
end

function dnl_at(c, P; phases_rad=zeros(3), dz=search_dz, nt=settings["Nt"], representation=:cp,
                phase=:validation)
    P in power_cases_W || throw(ArgumentError("Use one of the declared fixed powers."))
    phase in (:optimization,:validation) || error("Unknown budget bucket.")
    v = canonical_coefficients(c)
    p = canonical_phases(phases_rad)
    key = evaluation_key(v,P,dz,nt,representation;phases_rad=p)
    haskey(evaluation_cache,key) && return evaluation_cache[key]
    remaining(P,phase) > 0 || error("$(phase) evaluation budget reached at $(P/1000) kW")
    used[P][phase] += 1
    attempt_id[] += 1
    id = attempt_id[]
    try
        context = simulation_context(Int(nt),representation)
        measured = @timed begin
            input = gaussian_pulse(context.domain,3; peak_power=P,
                fwhm=settings["pulse_fwhm_ps"],coefficients=v .* cis.(p))
            linear_run = @timed solve_mmgnlse(input,context.linear,dz)
            nonlinear_run = @timed solve_mmgnlse(input,context.nonlinear,dz)
            linear = get_temporal_field(linear_run.value;z=:final)
            nonlinear = get_temporal_field(nonlinear_run.value;z=:final)
            all(isfinite,nonlinear) && all(isfinite,linear) || error("Nonfinite output field.")
            F_NL = field_fidelity(linear,nonlinear)
            photons = value(context.photons,nonlinear_run.value)
            photons12 = value(context.photons12,nonlinear_run.value)
            photons > 0 && isfinite(photons) || error("Invalid photon number.")
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
            field_cache[id] = (;input=copy(input),linear=copy(linear),nonlinear=copy(nonlinear))
            (; id, coefficient_1=v[1],coefficient_2=v[2],coefficient_3=v[3],
                phase_1_rad=p[1],phase_2_rad=p[2],phase_3_rad=p[3],
                power_W=Float64(P),dz=Float64(dz),nt=Int(nt),representation=string(representation),
                phase=string(phase),D_NL=1-F_NL,F_NL,
                D_NL_modes12=1-field_fidelity(linear,nonlinear;selected_modes=1:2,allow_empty=true),
                F_input=field_fidelity(input,nonlinear),photons,photons12,
                input_mode1_fraction=pin[1],input_mode2_fraction=pin[2],input_mode3_fraction=pin[3],
                mode1_energy_fraction=pout[1],mode2_energy_fraction=pout[2],mode3_energy_fraction=pout[3],
                output_input_energy_ratio=sum(output_energy)/sum(input_energy),
                time_edge_fraction=time_edge,frequency_edge_fraction=frequency_edge,
                linear_seconds=linear_run.time,nonlinear_seconds=nonlinear_run.time)
        end
        result = merge(measured.value,(;seconds=measured.time,allocated_bytes=measured.bytes))
        evaluation_cache[key] = result
        push!(evaluations,result)
        save_rows(joinpath(run_dir,"evaluations.csv"),evaluations)
        save_rows(joinpath(case_dir(P),"evaluations.csv"),filter(r->r.power_W==P,evaluations))
        @printf("%7.3f kW %s phi=%s DNL=%.8f | %s %.1f mm Nt=%d | %.2f s\n",
            P/1000,coeff_label(v),phase_label(p),result.D_NL,string(representation),1000dz,nt,result.seconds)
        flush(stdout)
        return result
    catch err
        push!(failures,(;id,power_W=P,coefficient_1=v[1],coefficient_2=v[2],coefficient_3=v[3],
            phase_1_rad=p[1],phase_2_rad=p[2],phase_3_rad=p[3],
            dz,nt,representation=string(representation),phase=string(phase),error=sprint(showerror,err)))
        save_rows(joinpath(run_dir,"failed_evaluations.csv"),failures)
        rethrow()
    end
end

function launch_gradient(q,P;h=fd_step,hphi=phase_fd_step,dz=search_dz,phase=:validation)
    new_probe_count(q,P;h,hphi,dz)<=remaining(P,phase) || error("Insufficient budget for a complete gradient.")
    g=zeros(5)
    for (j,c,p,w) in gradient_probes(q;h,hphi)
        g[j]+=w*dnl_at(c,P;phases_rad=p,dz,phase).D_NL
    end
    g
end

tangent_part(g)=vcat(g[1:3].-sum(g[1:3])/3,g[4:5])

function gradient_agreement(g,gh)
    a,b=tangent_part(g),tangent_part(gh)
    errors=abs.(a.-b)
    thresholds=gradient_check_atol .+ gradient_check_rtol .* max.(abs.(a),abs.(b))
    (;passed=all(errors.<=thresholds),errors,thresholds,max_ratio=maximum(errors./thresholds))
end

function history_row(iteration,row;g=fill(NaN,5),pg=NaN,step=0.0,phase_step=0.0,alpha=0.0)
    c=coefficients(row);p=phase_values(row)
    (;iteration,evaluation_id=row.id,power_W=row.power_W,
        coefficient_1=c[1],coefficient_2=c[2],coefficient_3=c[3],
        phase_1_rad=p[1],phase_2_rad=p[2],phase_3_rad=p[3],D_NL=row.D_NL,
        input_mode1_fraction=row.input_mode1_fraction,
        input_mode2_fraction=row.input_mode2_fraction,
        input_mode3_fraction=row.input_mode3_fraction,
        brightness_ratio=row.photons/preflights[row.power_W].fine.photons,
        previous_gradient_u1=g[1],previous_gradient_u2=g[2],previous_gradient_u3=g[3],
        previous_gradient_phi2=g[4],previous_gradient_phi3=g[5],
        previous_projected_gradient_norm=pg,max_coefficient_step=step,
        max_phase_step_rad=phase_step,backtracking_multiplier=alpha,
        optimization_evaluations=used[row.power_W][:optimization])
end

function optimize_launch(P)
    current=preflights[P].baseline
    q=controls(current)
    history=[history_row(0,current)]
    trials=NamedTuple[]
    status="iteration_budget"
    for iteration in 1:max_iterations
        if remaining(P,:optimization)<new_probe_count(q,P)+1
            status="evaluation_budget";break
        end
        g=launch_gradient(q,P;phase=:optimization)
        pg=norm(projected_gradient(q,g),Inf)
        if pg<=gradient_tolerance
            status="small_projected_gradient_search_grid";break
        end
        # The phase direction stays unwrapped in the local tangent space.
        direction=vcat(project_simplex(q[1:3].-learning_rate.*g[1:3]).-q[1:3],
            -learning_rate.*g[4:5])
        largest=100norm(direction[1:3],Inf)
        largest_phase=norm(direction[4:5],Inf)
        max(largest,largest_phase)>0 || (status="zero_projected_step";break)
        factor=min(1.0,largest>0 ? max_coefficient_change/largest : 1.0,
            largest_phase>0 ? max_phase_change_rad/largest_phase : 1.0)
        direction .*= factor
        accepted=false
        for attempt in 0:settings["max_backtracking_halvings"]
            alpha=0.5^attempt
            trial_q=project_controls(q.+alpha.*direction)
            c,p=unpack(trial_q)
            # Actual amplitude increment plus local phase increment before wrapping.
            delta=vcat(trial_q[1:3].-q[1:3],alpha.*direction[4:5])
            @assert isapprox(wrap_phase.(trial_q[4:5].-q[4:5]),delta[4:5];atol=1e-12)
            key=evaluation_key(c,P,search_dz,settings["Nt"],:cp;phases_rad=p)
            if !haskey(evaluation_cache,key) && remaining(P,:optimization)==0
                status="evaluation_budget";break
            end
            trial=dnl_at(c,P;phases_rad=p,phase=:optimization)
            step=100norm(delta[1:3],Inf);phase_step=norm(delta[4:5],Inf)
            feasible=trial.photons>=brightness_floors[P]
            slope=dot(g,delta)
            armijo=slope<0 && trial.D_NL<=current.D_NL+settings["armijo_c1"]*slope
            accepted=feasible && armijo && trial.D_NL<current.D_NL-minimum_decrease
            @assert step<=max_coefficient_change+1e-10 && phase_step<=max_phase_change_rad+1e-12
            push!(trials,(;iteration,attempt,evaluation_id=trial.id,
                coefficient_1=c[1],coefficient_2=c[2],coefficient_3=c[3],
                phase_1_rad=p[1],phase_2_rad=p[2],phase_3_rad=p[3],
                previous_DNL=current.D_NL,trial_DNL=trial.D_NL,
                gradient_dot_step=slope,alpha,max_coefficient_step=step,max_phase_step_rad=phase_step,
                feasible,armijo,accepted))
            save_rows(joinpath(case_dir(P),"line_search.csv"),trials)
            if accepted
                current=trial;q=controls(current)
                push!(history,history_row(iteration,current;g,pg,step,phase_step,alpha))
                save_rows(joinpath(case_dir(P),"history.csv"),history)
                @printf("ACCEPT %.3f kW iteration %d: c=%s phi=%s DNL=%.8f\n",
                    P/1000,iteration,coeff_label(coefficients(current)),phase_label(phase_values(current)),current.D_NL)
                flush(stdout);break
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
        "selected_coefficients"=>coefficients(current),"selected_phases_rad"=>phase_values(current),
        "selected_DNL_search"=>current.D_NL,"new_optimization_evaluations"=>used[P][:optimization]))
    (;best=current,history,trials,status)
end

function comparison_row(label,row,baseline)
    c=coefficients(row);p=phase_values(row)
    (;power_W=row.power_W,launch=label,coefficient_1=c[1],coefficient_2=c[2],coefficient_3=c[3],
        phase_1_rad=p[1],phase_2_rad=p[2],phase_3_rad=p[3],
        all_mode_DNL=row.D_NL,modes12_DNL_diagnostic=row.D_NL_modes12,
        brightness_ratio=row.photons/baseline.photons,dz_m=row.dz,nt=row.nt,
        representation=row.representation,evaluation_id=row.id)
end

function plot_modal_output(label,row)
    saved=field_cache[row.id]
    fields=(saved.input,saved.linear,saved.nonlinear)
    spectra=[fftshift(ifft(a,1),1) for a in fields]
    npzwrite(joinpath(case_dir(row.power_W),label*"_fields.npz"),Dict(
        "time_ps"=>times_ps,"wavelength_nm"=>wavelengths_nm,
        "input"=>saved.input,"linear"=>saved.linear,"nonlinear"=>saved.nonlinear,
        "power_W"=>row.power_W,"dz_m"=>row.dz,"launch_coefficients"=>coefficients(row),
        "launch_phases_rad"=>phase_values(row),"launch_complex_coefficients"=>coefficients(row).*cis.(phase_values(row)),
        "normalized_modal_weights"=>coefficients(row).*cis.(phase_values(row))./norm(coefficients(row))))
    caption=@sprintf("%s | %.3f kW | c=%s | phi=%s\nall-mode DNL %.6f | 1 mm",
        label,row.power_W/1000,coeff_label(coefficients(row)),phase_label(phase_values(row)),row.D_NL)
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

test_objective(c,p)=c[1]^2/sum(abs2,c)+0.2sin(p[2])+0.3cos(p[3])
