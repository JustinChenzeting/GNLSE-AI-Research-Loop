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

function wrap_phase(phi)
    y=mod(Float64(phi)+pi,2pi)-pi
    abs(y-pi)<1e-12 && return -Float64(pi)
    abs(y)<1e-14 ? 0.0 : y
end

function canonical_phases(phi)
    length(phi)==3 && all(isfinite,phi) || throw(ArgumentError("Supply three finite phases in radians."))
    p=wrap_phase.(Float64.(phi).-phi[1]);p[1]=0.0
    p
end

phase_values(row)=[row.phase_1_rad,row.phase_2_rad,row.phase_3_rad]

phase_label(phi)=@sprintf("[%.2f°, %.2f°, %.2f°]",rad2deg(phi[1]),rad2deg(phi[2]),rad2deg(phi[3]))

controls(row)=vcat(coefficients(row)./100,phase_values(row)[2:3])

function project_controls(q)
    length(q)==5 && all(isfinite,q) || throw(ArgumentError("Five finite controls required."))
    vcat(project_simplex(q[1:3]),wrap_phase.(q[4:5]))
end

unpack(q)=(100 .* q[1:3],vcat(0.0,q[4:5]))

# Phase derivatives live in the local angular tangent space: do not wrap a gradient.
projected_gradient(q,g)=vcat(q[1:3].-project_simplex(q[1:3].-g[1:3]),g[4:5])

coefficients(row) = [row.coefficient_1, row.coefficient_2, row.coefficient_3]

coeff_label(c) = @sprintf("[%.3f, %.3f, %.3f]", c[1], c[2], c[3])

budget(P, phase) = phase == :optimization ? max_optimizer_evaluations : max_check_evaluations

remaining(P, phase) = budget(P,phase)-used[P][phase]

function evaluation_key(c,P,dz,nt,representation;phases_rad=zeros(3))
    v=canonical_coefficients(c)
    p=canonical_phases(phases_rad)
    (Float64(P),Tuple(round.(v;digits=12)),Tuple(round.(p;digits=12)),Float64(dz),Int(nt),representation)
end

function coefficient_stencil(c;h=fd_step)
    lo,hi=coefficient_bounds
    lo<=c<=hi && 0<h<=(hi-lo)/2 || error("Invalid amplitude stencil.")
    if c-h>=lo && c+h<=hi
        ([c-h,c+h],[-1.0,1.0]./(2h))
    elseif c-h<lo
        ([c,c+h,c+2h],[-3.0,4.0,-1.0]./(2h))
    else
        ([c,c-h,c-2h],[3.0,-4.0,1.0]./(2h))
    end
end

function gradient_probes(q;h=fd_step,hphi=phase_fd_step)
    c,p=unpack(project_controls(q))
    probes=Tuple{Int,Vector{Float64},Vector{Float64},Float64}[]
    for j in 1:3
        points,weights=coefficient_stencil(c[j];h)
        for (point,weight) in zip(points,weights)
            v=copy(c);v[j]=point
            push!(probes,(j,v,copy(p),100weight))
        end
    end
    for j in 4:5, sign in (-1.0,1.0)
        phi=copy(p);phi[j-2]+=sign*hphi
        push!(probes,(j,copy(c),phi,sign/(2hphi)))
    end
    probes
end

function new_probe_count(q,P;h=fd_step,hphi=phase_fd_step,dz=search_dz)
    candidates=unique(evaluation_key(c,P,dz,settings["Nt"],:cp;phases_rad=p)
        for (_,c,p,_) in gradient_probes(q;h,hphi))
    count(k->!haskey(evaluation_cache,k),candidates)
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
