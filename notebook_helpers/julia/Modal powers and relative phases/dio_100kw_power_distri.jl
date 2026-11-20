function dio_at(x; dz=search_dz)
    x = canonical_x(x)
    key = cache_key(x,dz)
    haskey(evaluation_cache,key) && return evaluation_cache[key]
    filename = joinpath(cache_dir,key*".jls")
    if isfile(filename)
        saved = deserialize(filename)
        evaluation_cache[key] = saved.result
        field_cache[key] = saved.fields
        return saved.result
    end
    measured = @timed begin
        input = construct_input(x)
        nonlinear_run = @timed solve_mmgnlse(input, parameters_cp, dz)
        nonlinear_field = get_temporal_field(nonlinear_run.value; z=:final)
        F_IO = field_fidelity(input, nonlinear_field)
        D_IO = 1-F_IO
        LP01_photons = value(LP01_photon_observable, nonlinear_run.value)
        total_photons = value(all_photons, nonlinear_run.value)
        @assert isfinite(LP01_photons) && LP01_photons>0
        input_energy = vec(sum(abs2.(input); dims=(1,3)))
        output_energy = vec(sum(abs2.(nonlinear_field); dims=(1,3)))
        input_fractions = input_energy./sum(input_energy)
        output_fractions = output_energy./sum(output_energy)
        field_cache[key] = (; input=copy(input), nonlinear=copy(nonlinear_field))
        (; q2=x[1], q3=x[2], phi2_rad=x[3], phi3_rad=x[4],
            P1_W=fixed_mode1_power_W, P2_W=fixed_mode1_power_W*x[1], P3_W=fixed_mode1_power_W*x[2],
            total_power_W=sum(powers_from_x(x)), dz=Float64(dz), D_IO, F_IO, LP01_photons, total_photons,
            input_mode1_fraction=input_fractions[1], input_mode2_fraction=input_fractions[2],
            input_mode3_fraction=input_fractions[3], mode1_energy_fraction=output_fractions[1],
            mode2_energy_fraction=output_fractions[2], mode3_energy_fraction=output_fractions[3],
            output_input_energy_ratio=sum(output_energy)/sum(input_energy),
            nonlinear_seconds=nonlinear_run.time)
    end
    result = merge(measured.value, (;seconds=measured.time, allocated_bytes=measured.bytes))
    evaluation_cache[key]=result
    serialize(filename, (; result, fields=field_cache[key]))
    push!(evaluations,result)
    save_rows("evaluations_D_IO.csv",evaluations)
    @printf("P=[100, %.4f, %.4f] kW | phi=[0, %.5f, %.5f] | D_IO=%.8f | N_LP01=%.4e | dz=%.3f mm | %.1f s\n",
        result.P2_W/1000,result.P3_W/1000,x[3],x[4],result.D_IO,result.LP01_photons,dz*1000,result.seconds)
    flush(stdout)
    return result
end

function history_row(iteration,current; gradient=fill(NaN,4), step=zeros(4))
    return (;iteration, P1_W=current.P1_W,P2_W=current.P2_W,P3_W=current.P3_W,
        total_power_W=current.total_power_W,q2=current.q2,q3=current.q3,
        phi2_rad=current.phi2_rad,phi3_rad=current.phi3_rad,D_IO=current.D_IO,F_IO=current.F_IO,
        LP01_photons=current.LP01_photons,total_photons=current.total_photons,gradient_q2=gradient[1],gradient_q3=gradient[2],
        gradient_phi2=gradient[3],gradient_phi3=gradient[4],step_q2=step[1],step_q3=step[2],
        step_phi2=step[3],step_phi3=step[4])
end

function optimize_launch(;start_x=baseline_x)
    x=canonical_x(start_x); current=dio_at(x)
    @assert current.LP01_photons>=brightness_floor
    history=[history_row(0,current)]
    save_rows("optimization_history_D_IO.csv",history)
    evaluation_start=length(evaluations); started=time_ns()
    status="iteration budget reached"
    for iteration in 1:max_iterations
        gradient_keys=Set{String}()
        for j in 1:4
            points,_=derivative_stencil(x,j)
            for point in points
                !cache_exists(point,search_dz) && push!(gradient_keys,cache_key(point,search_dz))
            end
        end
        if length(evaluations)-evaluation_start+length(gradient_keys)+1>max_optimizer_evaluations
            status="objective evaluation budget reached"; break
        end
        g=launch_gradient(x)
        projected_gradient=[x[1]-clamp(x[1]-g[1],0.0,1.0),x[2]-clamp(x[2]-g[2],0.0,1.0),g[3],g[4]]
        if maximum(abs,projected_gradient)<gradient_tolerance
            status="projected gradient tolerance reached"; break
        end
        step=clamp.(-learning_rate.*g,-max_coordinate_changes,max_coordinate_changes)
        accepted=false; budget_hit=false
        for attempt in 1:12
            trial_x=project_x(x.+step)
            delta=coordinate_delta(trial_x,x)
            maximum(abs,delta)<1e-10 && break
            if !cache_exists(trial_x,search_dz) && length(evaluations)-evaluation_start>=max_optimizer_evaluations
                budget_hit=true; break
            end
            trial=dio_at(trial_x)
            if trial.LP01_photons>=brightness_floor && trial.D_IO<current.D_IO &&
                    trial.D_IO<=current.D_IO+1e-4*dot(g,delta)
                x,current=trial_x,trial
                push!(history,history_row(iteration,current;gradient=g,step=delta))
                save_rows("optimization_history_D_IO.csv",history)
                @printf("ACCEPT %2d: P=[100,%.6f,%.6f] kW; phi=[0,%.6f,%.6f]; D_IO=%.9f\n",
                    iteration,current.P2_W/1000,current.P3_W/1000,current.phi2_rad,current.phi3_rad,current.D_IO)
                flush(stdout); accepted=true; break
            end
            step./=2
        end
        if !accepted
            status=budget_hit ? "objective evaluation budget reached" : "line search failed"; break
        end
    end
    return (;best=current,history,status,seconds=(time_ns()-started)/1e9,
        new_objective_evaluations=length(evaluations)-evaluation_start)
end
