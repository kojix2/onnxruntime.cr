require "json"
require 'active_support/core_ext/string/inflections'

m = JSON.parse(File.read("capi.json"))
a = m.find_all {|i| i["name"] == "OrtApi"}.last
a["fields"].each do |i|
  puts i["name"].underscore
end
