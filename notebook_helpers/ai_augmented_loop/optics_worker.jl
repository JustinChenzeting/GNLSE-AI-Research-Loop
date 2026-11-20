# Persistent evaluator for the standalone Python loop. No notebook is executed.
using PulsePropagation, FFTW, LinearAlgebra, TOML, NPZ
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

function fidelity(a, b)
    size(a) == size(b) || error("Field grids differ")
    x, y = vec(@view(a[:, 1, :])), vec(@view(b[:, 1, :]))
    nx, ny = sum(abs2, x), sum(abs2, y)
    isfinite(nx) && isfinite(ny) && nx > 0 && ny > 0 || error("Nonfinite or zero LP01 norm")
    f = abs2(dot(x, y))/(nx*ny)
    isfinite(f) && -1e-12 <= f <= 1+1e-12 || error("Invalid normalized overlap")
    clamp(f, 0.0, 1.0)
end

# Preserve the reference notebook's objective-definition assertions.
let a = zeros(ComplexF64, 4, 3, 1)
    a[1,1,1] = 1
    @assert isapprox(fidelity(a,a), 1; atol=1e-12)
    @assert isapprox(fidelity(a,2exp(0.7im).*a), 1; atol=1e-12)
    b = copy(a); b[:,2,1] .= 3
    @assert isapprox(fidelity(a,b), 1; atol=1e-12)
    b .= 0; b[2,1,1] = 1
    @assert fidelity(a,b) == 0
    rejected = try fidelity(a, zero(a)); false catch; true end
    @assert rejected
end

const fibers = Dict{String,Any}()
const domains = Dict{Tuple{String,Int,Float64},Any}()
const baseline_photons = Dict{Tuple{String,Int,Float64,Float64},Float64}()

function setup_key(s)
    io = IOBuffer(); TOML.print(io, s; sorted=true); String(take!(io))
end

function evaluate_request(req)
    s, p, r = req["setup"], req["parameters"], req["resolution"]
    nt, dz = Int(r["Nt"]), Float64(r["dz_m"])
    nt >= 16 && iseven(nt) && dz > 0 || error("Invalid propagation resolution")
    key = setup_key(s)
    fiber = get!(fibers, key) do
        properties = compute_fiber_properties(
            StepIndex(s["carrier_m"], s["core_radius_m"], s["NA"]), Silica(), Int(s["num_modes"]);
            polarization=:scalar, mode_basis=:LP, beta_order=Int(s["beta_order"]))
        S = cp_compress(properties.S; error=s["cp_target_error"], seed=Int(s["cp_seed"]))
        (; properties, S)
    end
    gridkey = (key, nt, Float64(s["time_window_ps"]))
    d = get!(domains, gridkey) do
        grid = TimeGrid(nt, s["time_window_ps"])
        domain = MMGNLSEDomain(degrees_of_freedom(:time, :space), grid)
        prop, S = fiber.properties, fiber.S
        nonlinear = MMGNLSEParameters(domain; length=s["fiber_length_m"], beta=prop.beta, S,
            n2=prop.n2, omega0=prop.omega0, raman=prop.raman)
        linear = MMGNLSEParameters(domain; length=s["fiber_length_m"], beta=prop.beta, S,
            n2=0.0, omega0=prop.omega0, raman=prop.raman)
        detector = SpectralPhotonNumber(filter=ones(nt), modes=[1])
        all_detector = SpectralPhotonNumber(filter=ones(nt), modes=:all)
        (; grid, domain, nonlinear, linear, detector, all_detector)
    end
    if req["launch"] == "power"
        powers = [1000.0*p["power_kW"], 0., 0.]
        phases = zeros(3)
    else
        powers = [Float64(s["fixed_mode1_W"]), 1000.0*p["P2_kW"], 1000.0*p["P3_kW"]]
        phases = [0., p["phi2_rad"], p["phi3_rad"]]
    end
    all(isfinite, powers) && all(>=(0), powers) && sum(powers) > 0 || error("Invalid modal powers")
    input = gaussian_pulse(d.domain, Int(s["num_modes"]); peak_power=sum(powers),
        fwhm=s["pulse_fwhm_ps"], coefficients=sqrt.(powers) .* cis.(phases))
    started = time_ns()
    nonlinear_solution = solve_mmgnlse(input, d.nonlinear, dz)
    nonlinear = get_temporal_field(nonlinear_solution; z=:final)
    linear_solution = solve_mmgnlse(input, d.linear, dz)
    linear = get_temporal_field(linear_solution; z=:final)
    reference = req["objective"] == "D_IO" ? input : linear
    f = fidelity(reference, nonlinear)
    photons = value(d.detector, nonlinear_solution)
    total_photons = value(d.all_detector, nonlinear_solution)
    reference_solves = 0
    if req["brightness_reference"] == "matched_linear"
        reference_photons = value(d.detector, linear_solution)
    else
        reference_dz = Float64(s["brightness_reference_dz_m"])
        bkey = (key, nt, Float64(s["time_window_ps"]), reference_dz)
        if !haskey(baseline_photons, bkey)
            if req["launch"] == "power" && powers[1] == s["reference_power_W"] && dz == reference_dz
                baseline_photons[bkey] = photons
            else
                baseline = gaussian_pulse(d.domain, Int(s["num_modes"]); peak_power=s["reference_power_W"],
                    fwhm=s["pulse_fwhm_ps"], coefficients=[1.,0.,0.])
                baseline_photons[bkey] = value(d.detector, solve_mmgnlse(baseline, d.nonlinear, reference_dz))
                reference_solves += 1
            end
        end
        reference_photons = baseline_photons[bkey]
    end
    modal_energy = vec(sum(abs2.(nonlinear); dims=(1,3)))
    fractions = modal_energy ./ sum(modal_energy)
    energy_ratio = sum(modal_energy)/sum(abs2,input)
    valid = all(isfinite, nonlinear) && all(isfinite, linear) &&
        all(x -> isfinite(x) && x > 0, (photons, total_photons, reference_photons, energy_ratio)) &&
        all(x -> -1e-12 <= x <= 1+1e-12, fractions)
    artifact = joinpath(req["artifact_directory"], "fields.npz")
    npzwrite(artifact, Dict("input"=>input, "linear"=>linear, "nonlinear"=>nonlinear,
        "time_ps"=>collect(time_axis(d.grid)),
        "wavelength_nm"=>collect(wavelength_axis(d.grid, fiber.properties.omega0)).*1e9,
        "modal_power_W"=>powers, "phases_rad"=>phases, "dz_m"=>dz, "Nt"=>nt))
    Dict("status"=>"ok", "score"=>1-f,
        "checks"=>Dict("valid"=>valid, "overlap_algebra"=>true, "finite_positive_observables"=>valid,
                       "score_in_unit_interval"=>0 <= 1-f <= 1),
        "metrics"=>Dict("fidelity"=>f, "LP01_photons"=>photons, "total_photons"=>total_photons,
            "reference_photons"=>reference_photons, "brightness_ratio"=>photons/reference_photons,
            "output_input_energy_ratio"=>energy_ratio, "modal_energy_fractions"=>fractions,
            "total_peak_power_W"=>sum(powers), "cp_relative_error"=>fiber.S.relative_error,
            "nonlinear_solves"=>1+reference_solves, "linear_solves"=>1,
            "propagation_seconds"=>(time_ns()-started)/1e9))
end

for line in eachline(stdin)
    isempty(strip(line)) && continue
    req = TOML.parsefile(strip(line))
    result = try
        # Keep stdout exclusively for worker protocol messages.
        redirect_stdout(stderr) do
            evaluate_request(req)
        end
    catch err
        showerror(stderr, err, catch_backtrace()); println(stderr)
        Dict("status"=>"error", "error"=>sprint(showerror, err))
    end
    path = req["response_path"]
    open(path, "w") do io
        TOML.print(io, result)
    end
    println("RESULT\t", path); flush(stdout)
end
