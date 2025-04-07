# onnxruntime.cr 開発ガイドライン

## メモリ管理の重要ポイント

- **構造体ごとにメモリ確保方法を決定**: 各構造体について、Crystal側で確保するかC側で確保するかを事前に決める
- **混在を避ける**: 同じ構造体に対してCrystalとCの両方でメモリ確保が行われると二重解放の原因になる
- **解放状態の追跡**: `@resource_released`フラグを使用して解放状態を追跡する
- **エラー時のリソース解放**: `begin/ensure`ブロックを使用して確実にリソースを解放する

## 環境変数の管理

- **シングルトンパターン**: `OrtEnvironment`シングルトンを使用して環境変数を一元管理
- **スレッドセーフ**: ミューテックスを使用して環境変数へのアクセスを同期
- **明示的な解放**: 環境変数はファイナライザではなく明示的に解放する
- **使用パターン**:
  ```crystal
  # セッションの作成と使用
  session1 = OnnxRuntime::InferenceSession.new("model1.onnx")
  session2 = OnnxRuntime::InferenceSession.new("model2.onnx")
  
  # セッションの使用...
  
  # 終了時に明示的に解放
  session1.release_session
  session2.release_session
  
  # 最後に環境変数を解放
  OnnxRuntime::InferenceSession.release_env
  ```

## 構造体別の管理方法

### OrtEnv
- C API関数で作成、`api.release_env`で解放
- `OrtEnvironment`シングルトンで管理

### OrtSession
- C API関数で作成、`api.release_session`で解放
- `@session_released`フラグで解放状態を追跡

### OrtValue (Tensors)
- C API関数で作成、`api.release_value`で解放
- `ensure`ブロックで確実に解放

## よくある問題

1. Crystal側とC側の両方でリソースが解放される二重解放
2. エラー発生時にリソースが解放されないメモリリーク
3. 解放済みリソースへのアクセスによるダングリングポインタ
4. 共有リソースの同時解放による競合
