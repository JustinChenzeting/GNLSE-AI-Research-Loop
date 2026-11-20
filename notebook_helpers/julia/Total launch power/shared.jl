function save_rows(filename, rows)
    isempty(rows) && return
    open(joinpath(run_dir, filename), "w") do io
        println(io, join(string.(keys(first(rows))), ","))
        for row in rows
            println(io, join(values(row), ","))
        end
    end
end
