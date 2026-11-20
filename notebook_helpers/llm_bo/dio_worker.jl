# Each D_IO request runs one nonlinear solve; D_NL adds its matched linear solve.
using PulsePropagation, FFTW, LinearAlgebra, TOML, NPZ
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

function fidelity(a, b; selected_modes=1:1)
    size(a) == size(b) || error("Field grids differ")
    x, y = vec(@view(a[:,selected_modes,:])), vec(@view(b[:,selected_modes,:]))
    nx, ny = sum(abs2,x), sum(abs2,y)
    if !(isfinite(nx) && isfinite(ny) && nx > 0 && ny > 0)
        return nothing
    end
    f = abs2(dot(x,y))/(nx*ny)
    isfinite(f) && -1e-12 <= f <= 1+1e-12 || error("Invalid selected-mode overlap")
    clamp(f,0.,1.)
end

let a = zeros(ComplexF64,4,3,1)
    a[1,1,1] = 1
    @assert fidelity(a,a) == 1
    @assert isapprox(fidelity(a,2exp(0.7im).*a),1;atol=1e-12)
    b = copy(a); b[:,2,1] .= 8
    @assert fidelity(a,b) == 1
    @assert fidelity(a,b;selected_modes=1:3) < 1
    b .= 0; b[2,1,1] = 1
    @assert fidelity(a,b) == 0
    @assert isnothing(fidelity(a,zero(a)))
    b .= 0; b[1,2,1] = 1
    @assert isnothing(fidelity(b,b))
    @assert fidelity(b,b;selected_modes=1:3) == 1
end

const contexts = Dict{String,Any}()
function context(s, nt)
    io = IOBuffer(); TOML.print(io, Dict("setup"=>s,"Nt"=>nt);sorted=true)
    get!(contexts,String(take!(io))) do
        p = compute_fiber_properties(StepIndex(s["carrier_m"],s["core_radius_m"],s["NA"]),
            Silica(),Int(s["num_modes"]);polarization=:scalar,mode_basis=:LP,beta_order=Int(s["beta_order"]))
        S = cp_compress(p.S;error=s["cp_target_error"],seed=Int(s["cp_seed"]))
        grid = TimeGrid(nt,s["time_window_ps"])
        domain = MMGNLSEDomain(degrees_of_freedom(:time,:space),grid)
        nonlinear = MMGNLSEParameters(domain;length=s["fiber_length_m"],beta=p.beta,S,
            n2=p.n2,omega0=p.omega0,raman=p.raman)
        linear = MMGNLSEParameters(domain;length=s["fiber_length_m"],beta=p.beta,S,
            n2=0.0,omega0=p.omega0,raman=p.raman)
        detector = SpectralPhotonNumber(filter=ones(nt),modes=[1])
        all_detector = SpectralPhotonNumber(filter=ones(nt),modes=:all)
        (;p,S,grid,domain,nonlinear,linear,detector,all_detector)
    end
end

function evaluate_request(req)
    s,p,r = req["setup"],req["parameters"],req["resolution"]
    c = Float64[p["c1"],p["c2"],p["c3"]]
    all(v->isfinite(v) && 0<=v<=100,c) && isapprox(sum(c),100;atol=1e-6,rtol=0) || error("Invalid simplex point")
    haskey(p,"phi2_rad") == haskey(p,"phi3_rad") || error("Supply both relative phases")
    phases = Float64[0.0,get(p,"phi2_rad",0.0),get(p,"phi3_rad",0.0)]
    all(v->isfinite(v) && -pi<=v<=pi,phases) || error("Invalid relative phases")
    P, nt, dz = 1000.0*p["power_kW"],Int(r["Nt"]),Float64(r["dz_m"])
    adaptive = get(r,"adaptive_power_search",false)
    allowed = adaptive ? (isfinite(P) && P>0 && nt in (2048,4096)) :
                         (P in (100000.,81943.) && nt==2048)
    allowed && dz in (0.005,0.001) || error("Unexpected physics setup")
    adaptive && (get(req,"objective","D_IO") != "D_IO" || any(x -> !iszero(x),phases)) &&
        error("Adaptive power search uses zero-phase D_IO launches")
    d = context(s,nt)
    objective = get(req,"objective","D_IO")
    objective in ("D_IO","D_NL") || error("Unknown objective")
    is_dnl = objective == "D_NL"
    input = gaussian_pulse(d.domain,3;peak_power=P,fwhm=s["pulse_fwhm_ps"],coefficients=c .* cis.(phases))
    linear_run = is_dnl ? (@timed solve_mmgnlse(input,d.linear,dz)) : nothing
    linear = is_dnl ? get_temporal_field(linear_run.value;z=:final) : nothing
    is_dnl && !all(isfinite,linear) && error("Nonfinite linear field")
    measured = @timed solve_mmgnlse(input,d.nonlinear,dz)
    nonlinear = get_temporal_field(measured.value;z=:final)
    all(isfinite,nonlinear) || error("Nonfinite nonlinear field")
    f = fidelity(is_dnl ? linear : input,nonlinear;selected_modes=is_dnl ? (1:3) : (1:1))
    lp01_photons = value(d.detector,measured.value)
    total_photons = value(d.all_detector,measured.value)
    all(v->isfinite(v) && v>=0,(lp01_photons,total_photons)) || error("Invalid photon counts")
    photons = is_dnl ? total_photons : lp01_photons
    ein,eout = vec(sum(abs2.(input);dims=(1,3))),vec(sum(abs2.(nonlinear);dims=(1,3)))
    pin,pout = ein./sum(ein),eout./sum(eout)
    @assert isapprox(maximum(sum(abs2.(input);dims=(2,3))),P;rtol=1e-12)
    @assert isapprox(pin,c.^2 ./ sum(abs2,c);rtol=1e-12,atol=1e-14)
    tp = vec(sum(abs2.(nonlinear);dims=(2,3)))
    sp = vec(sum(abs2.(fftshift(ifft(nonlinear,1),1));dims=(2,3)))
    t,w = collect(time_axis(d.grid)),collect(frequency_axis(d.grid))
    fields = Dict(
        "input"=>input,"nonlinear"=>nonlinear,"time_ps"=>t,
        "wavelength_nm"=>collect(wavelength_axis(d.grid,d.p.omega0)).*1e9,
        "coefficients"=>c,"phases_rad"=>phases,"power_W"=>P,"dz_m"=>dz,"Nt"=>nt)
    is_dnl && (fields["linear"] = linear)
    npzwrite(joinpath(req["artifact_directory"],"fields.npz"),fields)
    defined = !isnothing(f)
    Dict("status"=>"ok","score"=>defined ? 1-f : 1.,
        "checks"=>Dict("objective_defined"=>defined,"valid"=>defined && photons>0),
        "metrics"=>Dict("LP01_photons"=>lp01_photons,"total_photons"=>total_photons,
            "input_power_fractions"=>pin,
            "output_energy_fractions"=>pout,"output_input_energy_ratio"=>sum(eout)/sum(ein),
            "time_edge_fraction"=>sum(tp[abs.(t).>0.45*s["time_window_ps"]])/sum(tp),
            "frequency_edge_fraction"=>sum(sp[abs.(w).>0.8*maximum(abs,w)])/sum(sp),
            "cp_rank"=>cp_rank(d.S),"cp_relative_error"=>d.S.relative_error,
            "nonlinear_solves"=>1,"linear_solves"=>Int(is_dnl),
            "nonlinear_seconds"=>measured.time,"linear_seconds"=>is_dnl ? linear_run.time : 0.0,
            "propagation_seconds"=>measured.time+(is_dnl ? linear_run.time : 0.0)))
end

function serve_requests(evaluator)
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        req = TOML.parsefile(strip(line))
        result = try
            redirect_stdout(stderr) do
                evaluator(req)
            end
        catch err
            showerror(stderr,err,catch_backtrace());println(stderr)
            Dict("status"=>"error","error"=>sprint(showerror,err))
        end
        open(req["response_path"],"w") do io
            TOML.print(io,result)
        end
        println("RESULT\t",req["response_path"]);flush(stdout)
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && serve_requests(evaluate_request)
