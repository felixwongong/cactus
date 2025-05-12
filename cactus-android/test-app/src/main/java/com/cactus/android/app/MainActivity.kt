package com.cactus.android.app

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.lifecycle.lifecycleScope
import com.cactus.android.LlamaContext
import com.cactus.android.LlamaInitParams
import com.cactus.android.LlamaCompletionParams
import com.cactus.android.app.databinding.ActivityMainBinding // Import ViewBinding class
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File 
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val TAG = "MainActivity"

    // Model download constants
    private val modelUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf"
    private val modelFileName = "QWEN2.5-1.5B-INST-Q8_0.gguf"
    private lateinit var modelFile: File // Will be initialized in onCreate

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize the target model file path in internal storage
        modelFile = File(filesDir, modelFileName)

        binding.buttonRunInference.setOnClickListener { runInference() }
        binding.editTextModelPath.setText(modelFile.absolutePath) // Show the target path
        binding.editTextModelPath.isEnabled = false // Disable editing the path for now

        // Start model download check when the activity is created
        downloadModelIfNeeded()
    }

    private fun downloadModelIfNeeded() {
        if (modelFile.exists()) {
            Log.i(TAG, "Model already exists at: ${modelFile.absolutePath}")
            binding.textViewOutput.text = "Model found locally. Enter prompt and run inference."
            binding.buttonRunInference.isEnabled = true
            return
        }

        Log.i(TAG, "Model not found. Starting download from: $modelUrl")
        binding.textViewOutput.text = "Downloading model ($modelFileName)... Please wait."
        binding.buttonRunInference.isEnabled = false // Disable button during download

        lifecycleScope.launch(Dispatchers.IO) { // Run download in background
            var success = false
            var errorMessage = ""
            try {
                val url = URL(modelUrl)
                val connection = url.openConnection() as HttpURLConnection
                connection.connect()

                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    throw RuntimeException("Server returned HTTP ${connection.responseCode} ${connection.responseMessage}")
                }

                val fileSize = connection.contentLength // Get file size for progress (optional)
                val inputStream: InputStream = connection.inputStream
                val outputStream = FileOutputStream(modelFile)
                val buffer = ByteArray(4096)
                var bytesRead: Int
                var totalBytesRead = 0L

                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead
                    // Optional: Calculate and display progress
                    val progress = if (fileSize > 0) (totalBytesRead * 100 / fileSize).toInt() else -1
                    withContext(Dispatchers.Main) {
                         binding.textViewOutput.text = "Downloading model... ${totalBytesRead / (1024*1024)} MB" + 
                                                      if(progress >= 0) " ($progress%)" else ""
                    }
                }

                outputStream.close()
                inputStream.close()
                connection.disconnect()
                Log.i(TAG, "Model download completed successfully.")
                success = true

            } catch (e: Exception) {
                Log.e(TAG, "Model download failed", e)
                errorMessage = e.message ?: "Unknown download error"
                // Attempt to delete partial file on error
                if (modelFile.exists()) {
                    modelFile.delete()
                }
            } finally {
                // Update UI on the main thread
                withContext(Dispatchers.Main) {
                    if (success) {
                        binding.textViewOutput.text = "Model downloaded. Ready for inference."
                        binding.buttonRunInference.isEnabled = true
                    } else {
                        binding.textViewOutput.text = "Model download failed: $errorMessage"
                        binding.buttonRunInference.isEnabled = false
                    }
                }
            }
        }
    }

    private fun runInference() {
        val modelPath = modelFile.absolutePath // Use the internal storage path
        val prompt = binding.editTextPrompt.text.toString().trim()

        if (!modelFile.exists()) {
            binding.textViewOutput.text = "Error: Model file not found at: $modelPath\nTry restarting the app to download."
            downloadModelIfNeeded() // Try downloading again
            return
        }
         if (prompt.isEmpty()) {
            Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
            return
        }

        binding.textViewOutput.text = "Loading model and running inference..."
        binding.buttonRunInference.isEnabled = false

        // Run model loading and inference in a background thread
        lifecycleScope.launch(Dispatchers.IO) {
            var resultText = "Error occurred"
            try {
                // --- Library Interaction --- 
                val initParams = LlamaInitParams(
                    modelPath = modelPath,
                    nCtx = 512 // Keep low for testing
                    // Add other basic params if needed, avoid LoRA/callbacks for now
                )
                
                // Use .use for automatic context.close()
                LlamaContext.create(initParams).use { context ->
                    Log.i(TAG, "LlamaContext created successfully.")
                    
                    val completionParams = LlamaCompletionParams(
                        temperature = 0.7f,
                        nPredict = 128 // Limit prediction length for testing
                        // Add other basic sampling params if needed
                    )

                    Log.i(TAG, "Running completion with prompt: '$prompt'")
                    // Call the native method - expect Map for now
                    val resultMap = LlamaContext.doCompletionNative(
                        context.contextPtr, // Access internal pointer (or make it public in library)
                        prompt,
                        completionParams.chatFormat,
                        completionParams.grammar ?: "",
                        completionParams.grammarLazy,
                        emptyList(), // Placeholder for grammarTriggers
                        emptyList(), // Placeholder for preservedTokens
                        completionParams.temperature,
                        completionParams.nThreads,
                        completionParams.nPredict,
                        completionParams.nProbs,
                        completionParams.penaltyLastN,
                        completionParams.penaltyRepeat,
                        completionParams.penaltyFreq,
                        completionParams.penaltyPresent,
                        completionParams.mirostat,
                        completionParams.mirostatTau,
                        completionParams.mirostatEta,
                        completionParams.topK,
                        completionParams.topP,
                        completionParams.minP,
                        completionParams.xtcThreshold,
                        completionParams.xtcProbability,
                        completionParams.typicalP,
                        completionParams.seed,
                        completionParams.stop?.toTypedArray() ?: emptyArray(),
                        completionParams.ignoreEos,
                        completionParams.logitBias ?: emptyMap(),
                        completionParams.dryMultiplier,
                        completionParams.dryBase,
                        completionParams.dryAllowedLength,
                        completionParams.dryPenaltyLastN,
                        completionParams.topNSigma,
                        completionParams.drySequenceBreakers?.toTypedArray() ?: emptyArray(),
                        null // No partial callback for now
                    )
                    
                    if (resultMap != null) {
                        Log.i(TAG, "Completion successful (raw map): $resultMap")
                        // Extract just the text for now, as LlamaCompletionResult parsing is TODO
                        resultText = resultMap["text"] as? String ?: "(No text found in result map)"
                    } else {
                         Log.e(TAG, "Completion failed: Native method returned null.")
                         resultText = "Completion failed: Native method returned null."
                    }
                }

            } catch (e: Throwable) {
                Log.e(TAG, "Error during Llama operation", e)
                resultText = "Error: ${e.message ?: e.toString()}"
            } finally {
                // Update UI back on the main thread
                withContext(Dispatchers.Main) {
                    binding.textViewOutput.text = resultText
                    binding.buttonRunInference.isEnabled = true
                }
            }
        }
    }

    // No need for onDestroy handling if using .use block
} 