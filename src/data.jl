module Data
using HTTP, CSV

export get_data

build_url(filename) = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/$(filename).csv"
retrieve_file(url) = HTTP.get(url).body |> IOBuffer |> CSV.read
get_data(filename) = build_url(filename) |> retrieve_file

end