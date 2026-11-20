wrap_phase(phi) = mod(phi+pi, 2pi)-pi

function canonical_x(x)
    length(x) == 4 && all(isfinite, x) || throw(ArgumentError("Four finite controls required."))
    0 <= x[1] <= 1 && 0 <= x[2] <= 1 || throw(ArgumentError("Power coordinates outside [0,1]."))
    return Float64[x[1], x[2], wrap_phase(x[3]), wrap_phase(x[4])]
end

powers_from_x(x) = [fixed_mode1_power_W, fixed_mode1_power_W*x[1], fixed_mode1_power_W*x[2]]

phases_from_x(x) = [0.0, x[3], x[4]]

function save_settings()
    open(joinpath(run_dir, "settings.toml"), "w") do io; TOML.print(io, settings); end
end

function construct_input(x)
    x = canonical_x(x)
    powers_W = powers_from_x(x)
    phases_rad = phases_from_x(x)
    return gaussian_pulse(domain, 3; peak_power=sum(powers_W),
        fwhm=settings["pulse_fwhm_ps"],
        coefficients=sqrt.(powers_W./powers_W[1]).*cis.(phases_rad))
end

cache_key(x, dz) = bytes2hex(sha256(string(physics_signature, repr(canonical_x(x)), repr(Float64(dz)))))

cache_exists(x, dz) = haskey(evaluation_cache, cache_key(x,dz)) || isfile(joinpath(cache_dir, cache_key(x,dz)*".jls"))

function save_rows(filename, rows)
    isempty(rows) && return
    open(joinpath(run_dir, filename), "w") do io
        println(io, join(string.(keys(first(rows))), ","))
        for row in rows; println(io, join(values(row), ",")); end
    end
end

function field_fidelity(reference, output)
    size(reference) == size(output) || throw(DimensionMismatch("Input/output field grids must match."))
    a = @view reference[:,1:1,:]
    b = @view output[:,1:1,:]
    na, nb = sum(abs2, a), sum(abs2, b)
    isfinite(na) && isfinite(nb) && na > 0 && nb > 0 || error("Invalid field norms.")
    f = abs2(dot(vec(a), vec(b))) / (na*nb)
    isfinite(f) && -1e-12 <= f <= 1+1e-12 || error("Invalid F_IO range.")
    return clamp(f, 0.0, 1.0)
end

x_from_row(row) = [row.q2,row.q3,row.phi2_rad,row.phi3_rad]

function derivative_stencil(x,j)
    h=fd_steps[j]
    if j<=2 && x[j]-h<0
        offsets=[0.0,h,2h]; weights=[-3.0,4.0,-1.0]./(2h)
    elseif j<=2 && x[j]+h>1
        offsets=[0.0,-h,-2h]; weights=[3.0,-4.0,1.0]./(2h)
    else
        offsets=[-h,h]; weights=[-1.0,1.0]./(2h)
    end
    points=[canonical_x(x .+ [k==j ? offset : 0.0 for k in 1:4]) for offset in offsets]
    return points,weights
end

function launch_gradient(x; dz=search_dz)
    g=zeros(4)
    for j in 1:4
        points,weights=derivative_stencil(x,j)
        g[j]=sum(w*dio_at(point;dz).D_IO for (point,w) in zip(points,weights))
    end
    return g
end

function project_x(x)
    canonical_x([clamp(x[1],0.0,1.0),clamp(x[2],0.0,1.0),x[3],x[4]])
end

function coordinate_delta(a,b)
    [a[1]-b[1],a[2]-b[2],wrap_phase(a[3]-b[3]),wrap_phase(a[4]-b[4])]
end

function save_summary()
    open(joinpath(run_dir,"summary.toml"),"w") do io; TOML.print(io,summary); end
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
    key=cache_key(x_from_row(row),row.dz)
    saved=field_cache[key]
    linear_path=joinpath(cache_dir,key*"_linear.jls")
    if isfile(linear_path)
        linear=deserialize(linear_path)
    else
        linear=get_temporal_field(solve_mmgnlse(saved.input,parameters_linear,row.dz);z=:final)
        serialize(linear_path,linear)
    end
    saved=merge(saved,(;linear))
    field_cache[key]=saved
    fields = (saved.input, saved.linear, saved.nonlinear)
    spectra = [fftshift(ifft(a, 1), 1) for a in fields]
    npzwrite(joinpath(run_dir, label*"_fields.npz"), Dict(
        "time_ps"=>times_ps, "wavelength_nm"=>wavelengths_nm,
        "input"=>saved.input, "linear"=>saved.linear, "nonlinear"=>saved.nonlinear,
        "powers_W"=>powers_from_x(x_from_row(row)), "phases_rad"=>phases_from_x(x_from_row(row)), "dz_m"=>row.dz,
        "launch_coefficients"=>sqrt.(powers_from_x(x_from_row(row))./fixed_mode1_power_W).*cis.(phases_from_x(x_from_row(row)))))
    caption = @sprintf("%s: P=[100, %.3f, %.3f] kW; phases=[0, %.4g, %.4g] rad; LP01 D_IO=%.6f; dz=1 mm",
        label,row.P2_W/1000,row.P3_W/1000,row.phi2_rad,row.phi3_rad,row.D_IO)
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
