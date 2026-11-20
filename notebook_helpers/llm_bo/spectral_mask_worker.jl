# Shared fiber construction, overlap definition and persistent request protocol.
include(joinpath(@__DIR__, "dio_worker.jl"))

function spectral_mask(omega, factors, halfwidth)
    length(factors)==100 || error("Exactly 100 factors required")
    all(v->isfinite(v) && 0<=v<=1 && abs(100v-round(100v))<1e-8,factors) ||
        error("Factors must lie on the 0.01 grid in [0,1]")
    mask=ones(Float64,length(omega)); bands=zeros(Int,length(omega))
    width=2halfwidth/100
    for k in eachindex(omega)
        # Small tolerance fixes floating-point ambiguity at internal boundaries.
        u=(omega[k]+halfwidth)/width
        b=floor(Int,u+1e-10)+1
        if 1<=b<=100 && omega[k]>=-halfwidth-1e-10 && omega[k]<halfwidth-1e-10
            bands[k]=b;mask[k]=factors[b]
        end
    end
    all(b->count(==(b),bands)>0,1:100) || error("The grid leaves an empty mask band")
    mask,bands
end

function evaluate_mask_request(req)
    s=copy(req["setup"]);p=req["parameters"];r=req["resolution"]
    nt=Int(r["Nt"]);dz=Float64(r["dz_m"]);T=Float64(r["time_window_ps"])
    s["time_window_ps"]=T
    nt in Int.(s["allowed_Nt"]) && T in s["allowed_windows_ps"] || error("Unexpected grid")
    dz in (0.005,0.001) || error("Unexpected step size")
    P=1000.0*p["power_kW"]; P in (100000.,81943.) || error("Unexpected power")
    factors=Float64.(p["factors"]);halfwidth=s["mask_halfwidth_rad_ps"]
    d=context(s,nt)
    original=gaussian_pulse(d.domain,3;peak_power=P,fwhm=s["pulse_fwhm_ps"],coefficients=ones(3))
    omega=collect(frequency_axis(d.grid));t=collect(time_axis(d.grid))
    mask,bands=spectral_mask(omega,factors,halfwidth)
    original_spectrum=fftshift(ifft(original,1),1)
    shaped_spectrum=original_spectrum .* reshape(mask,:,1,1)
    input=fft(ifftshift(shaped_spectrum,1),1) # No power/energy renormalization.
    all(isfinite,input) || error("Nonfinite masked input")
    original_energy=sum(abs2,original);input_energy=sum(abs2,input)
    input_energy>0 || error("Empty masked input")
    parseval_error=abs(input_energy-nt*sum(abs2,shaped_spectrum))/input_energy
    @assert parseval_error<1e-12
    @assert input_energy<=original_energy*(1+1e-12)
    @assert all(0 .<= mask .<= 1)
    input_power=vec(sum(abs2.(input);dims=(2,3)))
    if all(==(1.0),factors)
        @assert maximum(abs,input.-original)/sqrt(P)<1e-12
    end
    linear_run=@timed solve_mmgnlse(input,d.linear,dz)
    nonlinear_run=@timed solve_mmgnlse(input,d.nonlinear,dz)
    linear=get_temporal_field(linear_run.value;z=:final)
    nonlinear=get_temporal_field(nonlinear_run.value;z=:final)
    all(isfinite,linear) && all(isfinite,nonlinear) || error("Nonfinite output")
    f=fidelity(linear,nonlinear;selected_modes=1:3)
    photons=value(d.all_detector,nonlinear_run.value)
    lp01_photons=value(d.detector,nonlinear_run.value)
    isfinite(photons) && photons>0 || error("Invalid output photons")
    out_power=vec(sum(abs2.(nonlinear);dims=(2,3)))
    out_spectrum=vec(sum(abs2.(fftshift(ifft(nonlinear,1),1));dims=(2,3)))
    spectral_energy=vec(sum(abs2.(original_spectrum);dims=(2,3)))
    weights=[sum(spectral_energy[bands.==b])/sum(spectral_energy) for b in 1:100]
    input_center=sum(t.*input_power)/sum(input_power)
    fields=Dict("original_input"=>original,"input"=>input,"linear"=>linear,"nonlinear"=>nonlinear,
        "time_ps"=>t,"omega_detuning_rad_ps"=>omega,"omega_absolute_rad_ps"=>omega.+d.p.omega0,
        "wavelength_nm"=>collect(wavelength_axis(d.grid,d.p.omega0)).*1e9,
        "mask"=>mask,"factors"=>factors,"band_index"=>bands,"band_energy_fractions"=>weights,
        "band_edges_detuning_rad_ps"=>collect(range(-halfwidth,halfwidth;length=101)),
        "power_before_mask_W"=>P,"dz_m"=>dz,"Nt"=>nt,"time_window_ps"=>T)
    npzwrite(joinpath(req["artifact_directory"],"fields.npz"),fields)
    Dict("status"=>"ok","score"=>isnothing(f) ? 1.0 : 1-f,
        "checks"=>Dict("objective_defined"=>!isnothing(f),"valid"=>!isnothing(f),
                       "mask_bounds_and_grid"=>true,"parseval"=>parseval_error<1e-12),
        "metrics"=>Dict("total_photons"=>photons,"LP01_photons"=>lp01_photons,
            "input_energy_retained"=>input_energy/original_energy,
            "input_energy_nJ"=>input_energy*T/nt*1e-3,
            "original_energy_nJ"=>original_energy*T/nt*1e-3,
            "input_peak_kW"=>maximum(input_power)/1000,
            "input_rms_duration_ps"=>sqrt(sum((t.-input_center).^2 .* input_power)/sum(input_power)),
            "input_time_edge_fraction"=>sum(input_power[abs.(t).>0.45T])/sum(input_power),
            "time_edge_fraction"=>sum(out_power[abs.(t).>0.45T])/sum(out_power),
            "frequency_edge_fraction"=>sum(out_spectrum[abs.(omega).>0.8maximum(abs,omega)])/sum(out_spectrum),
            "output_input_energy_ratio"=>sum(out_power)/input_energy,
            "band_energy_fractions"=>weights,"outside_band_energy_fraction"=>sum(spectral_energy[bands.==0])/sum(spectral_energy),
            "band_sample_counts"=>[count(==(b),bands) for b in 1:100],
            "parseval_relative_error"=>parseval_error,
            "cp_rank"=>cp_rank(d.S),"cp_relative_error"=>d.S.relative_error,
            "linear_seconds"=>linear_run.time,"nonlinear_seconds"=>nonlinear_run.time,
            "propagation_seconds"=>linear_run.time+nonlinear_run.time,
            "linear_solves"=>1,"nonlinear_solves"=>1))
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && serve_requests(evaluate_mask_request)
