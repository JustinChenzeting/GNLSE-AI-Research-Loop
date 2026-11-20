# Created a Gaussian and simulates it passing through the fibre
function squeezing_at(power_W; dz=search_dz)
    key = (Float64(power_W), Float64(dz))
    haskey(evaluation_cache, key) && return evaluation_cache[key]
    power_W > 0 || throw(ArgumentError("Power must be positive."))
    cpu_start = ccall(:clock, Int64, ()) / 1e6
    measured = @timed begin
        input = gaussian_pulse(domain, settings["num_modes"];
            peak_power=power_W, fwhm=settings["pulse_fwhm_ps"],
            coefficients=[1.0, 0.0, 0.0])
        # Simulate input pulse travelling through the fibre
        forward = solve_mmgnlse(input, parameters_cp, dz)
        # Counting photons
        photons = value(photon_observable, forward)
        terminal = terminal_condition(photon_observable, forward)
        backward = solve_adjoint(terminal, parameters_cp, dz;
            forward_solution=forward, adjoint_mode=:discrete_rk4ip,
            checkpoint_stride=32)
        sensitivity = get_spectral_field(backward; z=:initial, units=:photon)
        variance = sum(abs2, sensitivity)
        fano = variance / photons
        @assert isfinite(fano) && fano > 0 && photons > 0
        (; power_W=Float64(power_W), dz=Float64(dz), fano,
            squeezing_dB=10log10(fano), photons,
            photon_transmission=photons/value(all_photons, forward))
    end
    result = merge(measured.value, (; seconds=measured.time,
        cpu_seconds=ccall(:clock, Int64, ())/1e6-cpu_start,
        allocated_bytes=measured.bytes))
    evaluation_cache[key] = result
    push!(evaluations, result)
    save_rows("evaluations.csv", evaluations)
    @printf("P=%8.3f kW | J=%9.5f dB | F=%.6f | N=%.4e | dz=%.3g m | %.1f s\n",
        power_W/1000, result.squeezing_dB, result.fano, result.photons, dz, result.seconds)
    flush(stdout)
    return result
end

# Compute the gradient of the squeezing with respect to the power using finite differences
function power_gradient(u; h=fd_step, dz=search_dz)
    plus = squeezing_at(reference_power_W*(u+h); dz)
    minus = squeezing_at(reference_power_W*(u-h); dz)
    return (plus.squeezing_dB-minus.squeezing_dB)/(2h)
end

function optimize_power(; start_power_W=reference_power_W)
    lower, upper = power_bounds_W ./ reference_power_W
    u = clamp(start_power_W/reference_power_W, lower, upper)
    current = squeezing_at(u*reference_power_W)
    @assert current.photons >= brightness_floor
    history = [(iteration=0, power_W=current.power_W,
        squeezing_dB=current.squeezing_dB, fano=current.fano,
        photons=current.photons, gradient_used=NaN, step_u=0.0)]
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
            trial = squeezing_at(trial_u*reference_power_W)
            if trial.photons >= brightness_floor &&
                    trial.squeezing_dB <= current.squeezing_dB + 1e-4*g*actual_step
                u, current = trial_u, trial
                push!(history, (; iteration, power_W=current.power_W,
                    squeezing_dB=current.squeezing_dB, fano=current.fano,
                    photons=current.photons, gradient_used=g, step_u=actual_step))
                save_rows("optimization_history.csv", history)
                @printf("ACCEPT %2d: %.3f kW, %.6f dB\n", iteration, current.power_W/1000, current.squeezing_dB)
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
