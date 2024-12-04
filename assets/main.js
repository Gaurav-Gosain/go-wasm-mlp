const go = new Go(); // Defined in wasm_exec.js
const WASM_URL = "tiny.wasm";

var wasm;

document.getElementById("status").innerText = "Initializing wasm...";

/**
 * Initializes the wasm module and starts the go program
 * @param  {WebAssembly.WebAssemblyInstantiatedSource}  result
 * @returns
 */
const initWasm = async (result) => {
  // TODO: figure out how to use the wasm module
  //
  // mod = result.module;
  //

  wasm = result.instance;
  document.getElementById("status").innerText = "Initialization complete.";
  go.run(wasm);
};

if ("instantiateStreaming" in WebAssembly) {
  WebAssembly.instantiateStreaming(fetch(WASM_URL), go.importObject).then(
    initWasm,
  );
} else {
  fetch(WASM_URL)
    .then((resp) => resp.arrayBuffer())
    .then((bytes) =>
      WebAssembly.instantiate(bytes, go.importObject).then(initWasm),
    );
}
