require "json"
require 'active_support/core_ext/string/inflections'

m = JSON.parse(File.read(File.join(__dir__, "capi.json")))
a = m.find_all {|i| i["name"] == "OrtApi"}.last
a["fields"].each do |i|
  puts i["name"].underscore
end
