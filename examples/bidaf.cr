require "onnxruntime"

# https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/bidirectional_attention_flow
model = OnnxRuntime::Model.new("./src/models/bidaf-9.onnx")

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

# context = "A quick brown fox jumps over the lazy dog."
# query = "What color is the fox?"

context = "Although Alan Turing made numerous contributions to mathematics and computer science, it was John von Neumann who formalized the architecture used in most modern computers. Turing, however, is best known for his work on breaking the Enigma code during World War II."
query = "Who developed the foundational architecture of modern computers?"

puts "Context: #{context}"
puts "Query: #{query}"
answer = predict_w(model, context, query)
# Answer: john von neumann
puts "Answer: #{answer}"

model.release
OnnxRuntime::InferenceSession.release_env
