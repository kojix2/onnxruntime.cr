require "../src/onnxruntime"
require "option_parser"

model_path = ""
context = "Although Alan Turing made numerous contributions to mathematics and computer science, it was John von Neumann who formalized the architecture used in most modern computers. Turing, however, is best known for his work on breaking the Enigma code during World War II."
query = "Who developed the foundational architecture of modern computers?"

OptionParser.parse do |parser|
  parser.banner = <<-BANNER
    Usage: bidaf [options]
    Bidirectional Attention Flow (BiDAF) for Question Answering
    Download bidaf-9.onnx from https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/bidirectional_attention_flow
      bidaf -m bidaf-9.onnx
      bidaf -m bidaf-9.onnx -c "A quick brown fox jumps over the lazy dog." -q "What color is the fox?"
    Options:
  BANNER

  parser.on("-m MODEL", "--model=MODEL", "Path to the ONNX model file (required)") do |m|
    model_path = m
  end

  parser.on("-c CONTEXT", "--context=CONTEXT", "The context text for question answering") do |c|
    context = c
  end

  parser.on("-q QUERY", "--query=QUERY", "The question to answer") do |q|
    query = q
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit(0)
  end

  parser.invalid_option do |flag|
    STDERR.puts "ERROR: #{flag} is not a valid option."
    STDERR.puts parser
    exit(1)
  end

  parser.missing_option do |flag|
    STDERR.puts "ERROR: Missing required option: #{flag}"
    STDERR.puts parser
    exit(1)
  end
end

if model_path.empty?
  STDERR.puts "ERROR: Model path is required. Use --model=MODEL or -m MODEL"
  exit(1)
end

begin
  model = OnnxRuntime::Model.new(model_path)
rescue ex
  STDERR.puts "ERROR: Failed to load model: #{ex.message}"
  exit(1)
end

private def tokenize(text : String) : Array(String)
  text.scan(/\w+|[^\s\w]/).map(&.[0])
end

private def preprocess(text : String)
  tokens = tokenize(text)
  words = tokens.map { |token| [token.downcase] }

  chars = tokens.flat_map do |token|
    padded = token.chars.map(&.to_s)[0, 16]
    padded + Array(String).new(16 - padded.size, "")
  end

  {words, chars, tokens}
end

def predict_w(model, context : String, query : String)
  cw, cc, _context_tokens = preprocess(context)
  qw, qc, _ = preprocess(query)
  inputs = {
    "context_word" => cw.map(&.first),
    "context_char" => cc,
    "query_word"   => qw.map(&.first),
    "query_char"   => qc,
  }

  shape = {
    "context_word" => [cw.size.to_i64, 1_i64],
    "context_char" => [cc.size.to_i64 // 16, 1_i64, 1_i64, 16_i64],
    "query_word"   => [qw.size.to_i64, 1_i64],
    "query_char"   => [qc.size.to_i64 // 16, 1_i64, 1_i64, 16_i64],
  }
  answer = model.predict(inputs, nil, shape: shape)

  cw.map(&.first)[
    answer["start_pos"].as(Array(Int32)).first..answer["end_pos"].as(Array(Int32)).first,
  ]
    .join(" ")
end

puts "Context: #{context}"
puts "Query: #{query}"
answer = predict_w(model, context, query)
# Answer: john von neumann
puts "Answer: #{answer}"

model.release
OnnxRuntime::InferenceSession.release_env
