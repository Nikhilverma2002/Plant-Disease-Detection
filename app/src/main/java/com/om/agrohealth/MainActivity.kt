package com.om.agrohealth // Make sure this package name matches your project structure

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.Window
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.IOException // Added for file descriptor closing
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
// import kotlin.experimental.toUByte // May not be needed depending on Kotlin version

class MainActivity : AppCompatActivity() {

    // --- Constants and Configuration ---
    companion object {
        private const val TAG = "AgroHealthApp" // Define TAG for logging
        private const val REQUEST_CODE_STORAGE_PERMISSION = 1
        private const val CAMERA_REQUEST = 1888
        private const val MY_CAMERA_PERMISSION_CODE = 100
        private const val PICK_FILE = 1
        private const val INPUT_IMG_WIDTH = 224 // Model Input Width - **VERIFY if new model requires different size**
        private const val INPUT_IMG_HEIGHT = 224 // Model Input Height - **VERIFY if new model requires different size**
        private const val CHANNEL_SIZE = 3 // RGB
        // *** Model expects UINT8 (Byte) input *** - **VERIFY if new model requires FLOAT32 etc.**
        private const val BYTES_PER_CHANNEL = 1
        private const val MODEL_INPUT_SIZE = BYTES_PER_CHANNEL * INPUT_IMG_WIDTH * INPUT_IMG_HEIGHT * CHANNEL_SIZE
    }

    private var selected_uri_image = Uri.parse("")

    // --- NEW Labels corresponding to NEW model output indices ---
    // **** UPDATED LABELS START ****
    private val labels = listOf(
        "Rice rusted bacterial smut",   // Index 0
        "Potato Late/Early Blight",     // Index 1
        "Tomato Powdery Mildew",        // Index 2
        "Tomato Gray/Bacterial spot",  // Index 3
        "Pumpkin Mosaic Disease",       // Index 4
        "Pumpkin Powdery Mildew",       // Index 5
        "Eggplant (Brinjal) Mosaic Virus" // Index 6
        // Previous labels removed
    )
    // **** UPDATED LABELS END ****

    // --- Output Array: Changed to ByteArray to match UINT8 model output ---
    // The size will be automatically determined by the new `labels.size` (which is now 7)
    // ** VERIFY if new model outputs FLOAT32 probabilities instead of UINT8 scores **
    private val predictionResultArray = Array(1) { ByteArray(labels.size) }

    // TFLite Interpreter (Lazy Initialization)
    private val interpreter: Interpreter? by lazy {
        loadModelFile()?.let {
            try {
                // ** Add Interpreter Options here if needed (e.g., for GPU delegate, NNAPI) **
                // val options = Interpreter.Options()
                // options.setNumThreads(4) // Example: Set number of threads
                // Interpreter(it, options)
                Interpreter(it)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize Interpreter", e)
                Toast.makeText(applicationContext, "Error initializing model interpreter.", Toast.LENGTH_LONG).show()
                null // Return null if init fails
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main) // Ensure you have activity_main.xml layout

        // --- Window Styling ---
        setupWindowStyling()

        // --- Button Listeners ---
        setupButtonClickListeners() // Ensure R.id.textView2 and R.id.pic exist in your layout

        // --- Initial Check ---
        if (interpreter == null) {
            Log.e(TAG, "TensorFlow Lite Interpreter failed to initialize in lazy block.")
            // Consider disabling buttons if interpreter is null
            try {
                findViewById<TextView>(R.id.prediction).text = "Model failed to load." // Ensure R.id.prediction exists
            } catch (e: Exception) {
                Log.e(TAG, "Failed to find prediction TextView", e)
            }
        } else {
            Log.d(TAG, "TensorFlow Lite Interpreter initialized successfully.")
        }
    }

    // --- Setup Functions (for organization) ---
    private fun setupWindowStyling() {
        val window: Window = this.window
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
        window.statusBarColor = ContextCompat.getColor(this, R.color.white) // Ensure R.color.white exists
        window.navigationBarColor = ContextCompat.getColor(this, R.color.white) // Ensure R.color.white exists
    }

    private fun setupButtonClickListeners() {
        // Ensure these IDs exist in your activity_main.xml layout file
        try {
            findViewById<TextView>(R.id.textView2).setOnClickListener { // Select File Button ID
                checkStoragePermissionAndSelect()
            }

            findViewById<TextView>(R.id.pic).setOnClickListener { // Take Picture Button ID
                checkCameraPermissionAndOpen()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up button listeners. Check layout IDs (R.id.textView2, R.id.pic)", e)
            Toast.makeText(this, "Error finding UI elements.", Toast.LENGTH_LONG).show()
        }
    }

    // --- Permission Handling ---
    private fun checkStoragePermissionAndSelect() {
        if (ContextCompat.checkSelfPermission(
                applicationContext,
                Manifest.permission.READ_EXTERNAL_STORAGE // Use READ_MEDIA_IMAGES on Android 13+ if targeting API 33+
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), // Use READ_MEDIA_IMAGES on Android 13+ if targeting API 33+
                REQUEST_CODE_STORAGE_PERMISSION
            )
        } else {
            selectImageFile()
        }
    }

    private fun checkCameraPermissionAndOpen() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), MY_CAMERA_PERMISSION_CODE)
        } else {
            openCamera()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isEmpty()) {
            Log.w(TAG, "Permission result cancelled or empty.")
            return // Nothing to do if grant results are empty
        }

        when (requestCode) {
            MY_CAMERA_PERMISSION_CODE -> {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
                    openCamera()
                } else {
                    Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
                }
            }
            REQUEST_CODE_STORAGE_PERMISSION -> {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Storage permission granted", Toast.LENGTH_SHORT).show()
                    selectImageFile()
                } else {
                    Toast.makeText(this, "Storage permission denied", Toast.LENGTH_LONG).show()
                }
            }
            // Handle other permission requests if needed
        }
    }

    // --- Intent Actions ---
    private fun openCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        // Ensure there's an activity to handle the intent
        if (cameraIntent.resolveActivity(packageManager) != null) {
            startActivityForResult(cameraIntent, CAMERA_REQUEST)
        } else {
            Toast.makeText(this, "No camera app found", Toast.LENGTH_SHORT).show()
        }
    }

    private fun selectImageFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "image/*" // Limit to images
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        try {
            startActivityForResult(
                Intent.createChooser(intent, "Select Picture"),
                PICK_FILE
            )
        } catch (ex: android.content.ActivityNotFoundException) {
            // Potentially direct user to install a file manager or handle differently
            Toast.makeText(this, "Please install a File Manager.", Toast.LENGTH_SHORT).show()
        }
    }

    // --- Model Loading ---
    private fun loadModelFile(): MappedByteBuffer? {
        val modelFileName = "model.tflite" // Ensure this matches the file in your assets folder
        return try {
            val fileDescriptor: AssetFileDescriptor = assets.openFd(modelFileName)
            Log.d(TAG, "Model file '$modelFileName' found in assets.")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel: FileChannel = inputStream.channel
            val startOffset: Long = fileDescriptor.startOffset
            val declaredLength: Long = fileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            // It's good practice to close streams and descriptors, though MappedByteBuffer might keep the file open
            inputStream.close()
            fileDescriptor.close()
            Log.d(TAG, "Model file loaded into MappedByteBuffer.")
            mappedByteBuffer
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model file '$modelFileName' from assets", e)
            Toast.makeText(this, "Failed to load model file. Check assets folder.", Toast.LENGTH_LONG).show()
            null
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error loading model file '$modelFileName'", e)
            Toast.makeText(this, "Failed to load model file.", Toast.LENGTH_LONG).show()
            null
        }
    }

    // --- Bitmap to ByteBuffer Conversion (Quantized UInt8 Input) ---
    // This version prepares input for a model expecting Bytes (0-255)
    // ** MODIFY THIS if your NEW model expects FLOAT32 input **
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        // Ensure bitmap is already resized to INPUT_IMG_WIDTH x INPUT_IMG_HEIGHT before calling this
        if (bitmap.width != INPUT_IMG_WIDTH || bitmap.height != INPUT_IMG_HEIGHT) {
            Log.e(TAG, "Bitmap size mismatch! Expected ${INPUT_IMG_WIDTH}x${INPUT_IMG_HEIGHT}, got ${bitmap.width}x${bitmap.height}")
            // Handle error appropriately, maybe throw an exception or return null/empty buffer
            // For now, proceed, but results will be incorrect. Best practice is to resize first.
            // Consider resizing here if it wasn't done before, but resizing in runInference is safer.
            // val resized = Bitmap.createScaledBitmap(bitmap, INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT, true)
            // Use 'resized' below instead of 'bitmap' if you resize here.
        }

        val byteBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder()) // Use native byte order

        val pixels = IntArray(INPUT_IMG_WIDTH * INPUT_IMG_HEIGHT)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        // Iterate assuming HWC format for the model input buffer
        for (i in 0 until INPUT_IMG_HEIGHT) { // Rows (Height)
            for (j in 0 until INPUT_IMG_WIDTH) { // Columns (Width)
                val pixelVal = pixels[pixel++]
                // Extract RGB bytes and put them into the buffer in RGB order
                byteBuffer.put((pixelVal shr 16 and 0xFF).toByte()) // Red
                byteBuffer.put((pixelVal shr 8 and 0xFF).toByte())  // Green
                byteBuffer.put((pixelVal and 0xFF).toByte())         // Blue
            }
        }
        // Log.d(TAG, "Bitmap converted to ByteBuffer (UInt8). Size: ${byteBuffer.position()}") // Optional log
        return byteBuffer
    }


    // --- Activity Result Handling ---
    @Deprecated("Deprecated in Java") // This annotation is correct for overriding onActivityResult
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK && data != null) {
            when (requestCode) {
                PICK_FILE -> {
                    selected_uri_image = data.data ?: Uri.EMPTY // Use Uri.EMPTY for null safety
                    if (selected_uri_image != Uri.EMPTY) {
                        Log.d(TAG, "Image selected from gallery: $selected_uri_image")
                        processImage(selected_uri_image)
                    } else {
                        Log.w(TAG, "Failed to get image URI from gallery selection.")
                        Toast.makeText(this, "Failed to get image URI", Toast.LENGTH_SHORT).show()
                    }
                }
                CAMERA_REQUEST -> {
                    // Camera intent might return thumbnail in "data" extra OR save to a predefined URI
                    // This code handles the thumbnail case. For full-res images, you'd typically provide a URI via MediaStore.EXTRA_OUTPUT.
                    val thumbnail = data.extras?.get("data") as? Bitmap
                    if (thumbnail != null) {
                        Log.d(TAG, "Image captured from camera (thumbnail).")
                        processCameraImage(thumbnail)
                    } else {
                        Log.w(TAG, "Failed to get camera image data (thumbnail) from extras.")
                        // Check if data.data contains a URI (less common for standard ACTION_IMAGE_CAPTURE without EXTRA_OUTPUT)
                        if (data.data != null) {
                            Log.d(TAG, "Camera might have returned URI: ${data.data}")
                            processImage(data.data!!) // Process if URI exists
                        } else {
                            Toast.makeText(this, "Failed to get camera image", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        } else if (resultCode != RESULT_CANCELED) {
            Log.w(TAG, "onActivityResult failed or cancelled. ResultCode: $resultCode")
            Toast.makeText(this, "Activity failed or was cancelled", Toast.LENGTH_SHORT).show()
        } else {
            Log.d(TAG, "onActivityResult cancelled by user.") // User pressed back or cancelled selection
        }
    }

    // --- Image Processing Logic ---
    private fun processImage(imageUri: Uri) {
        var bitmap: Bitmap? = null
        try {
            bitmap = getBitmapFromUri(imageUri)
            if (bitmap != null) {
                // Optionally display the selected image:
                // findViewById<ImageView>(R.id.your_image_view_id).setImageBitmap(bitmap) // Ensure you have an ImageView with this ID
                Log.d(TAG, "Bitmap loaded from URI, proceeding to inference.")
                runInference(bitmap)
            } else {
                Log.e(TAG, "Failed to load bitmap from URI: $imageUri")
                Toast.makeText(this, "Failed to load image from storage", Toast.LENGTH_SHORT).show()
                try { findViewById<TextView>(R.id.prediction).text = "Error loading image" } catch (e: Exception) {}
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image URI: $imageUri", e)
            Toast.makeText(this, "Error processing selected image", Toast.LENGTH_SHORT).show()
            try { findViewById<TextView>(R.id.prediction).text = "Error processing image" } catch (e: Exception) {}
        } finally {
            // Avoid recycling the original bitmap here if it might be displayed elsewhere
            // Recycling should happen after inference if a copy/resized version was made.
        }
    }

    private fun processCameraImage(bitmap: Bitmap) {
        try {
            // Optionally display the captured image:
            // findViewById<ImageView>(R.id.your_image_view_id).setImageBitmap(bitmap)
            Log.d(TAG, "Camera bitmap received, proceeding to inference.")
            runInference(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error processing camera image", e)
            Toast.makeText(this, "Error processing camera image", Toast.LENGTH_SHORT).show()
            try { findViewById<TextView>(R.id.prediction).text = "Error processing image" } catch (e: Exception) {}
        }
        // Note: The bitmap passed here is often a thumbnail. Don't recycle it if displaying it.
    }


    // --- Inference Execution ---
    private fun runInference(bitmap: Bitmap) {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter is null, cannot run inference.")
            Toast.makeText(this, "Model interpreter not ready", Toast.LENGTH_SHORT).show()
            try { findViewById<TextView>(R.id.prediction).text = "Model not loaded" } catch (e: Exception) {}
            return
        }

        Log.d(TAG, "Starting inference...")
        var resizedBitmap: Bitmap? = null
        try {
            // 1. Resize the bitmap to the model's expected input size
            // It's crucial that this matches INPUT_IMG_WIDTH and INPUT_IMG_HEIGHT
            resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT, true)
            Log.d(TAG, "Input Bitmap resized to ${resizedBitmap.width}x${resizedBitmap.height}")

            // 2. Convert the *resized* bitmap to ByteBuffer (UINT8 format)
            // ** MODIFY convertBitmapToByteBuffer if your model needs FLOAT32 **
            val modelInput = convertBitmapToByteBuffer(resizedBitmap)
            modelInput.rewind() // Ensure buffer position is at the beginning

            // Clear the previous results before running inference
            predictionResultArray[0].fill(0) // Zero out the byte array

            // 3. Run inference
            // ** MODIFY predictionResultArray type if model outputs FLOAT32 **
            Log.d(TAG, "Running interpreter.run...")
            interpreter?.run(modelInput, predictionResultArray)
            Log.d(TAG, "Interpreter.run completed.")

            // 4. Process and display results (Pass the ByteArray containing UINT8 scores)
            // ** MODIFY displayPrediction if model outputs FLOAT32 **
            displayPrediction(predictionResultArray[0])

        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "Error running inference: Check input/output dimensions or types.", e)
            Toast.makeText(this, "Inference error: Model incompatibility?", Toast.LENGTH_SHORT).show()
            try { findViewById<TextView>(R.id.prediction).text = "Error: Model mismatch?" } catch (ex: Exception) {}
        }
        catch (e: Exception) {
            // Log the specific error
            Log.e(TAG, "Error running model inference", e)
            Toast.makeText(this, "Inference failed. Check logs.", Toast.LENGTH_SHORT).show()
            try { findViewById<TextView>(R.id.prediction).text = "Error during prediction" } catch (ex: Exception) {}
        } finally {
            // Recycle the resized bitmap *if* it's different from the original input bitmap
            // and if the original bitmap isn't needed anymore (e.g., not displayed).
            // Be cautious recycling bitmaps displayed in ImageViews.
            if (resizedBitmap != null && resizedBitmap != bitmap && !resizedBitmap.isRecycled) {
                // resizedBitmap.recycle() // Uncomment cautiously if bitmap management requires it
                Log.d(TAG,"Resized bitmap could be recycled here.")
            }
        }
    }


    // --- Prediction Display Function (Handles ByteArray from UINT8 model output) ---
    // ** MODIFY THIS if your NEW model outputs FLOAT32 probabilities **
    private fun displayPrediction(scores: ByteArray) {
        if (scores.isEmpty()) {
            Log.e(TAG, "Prediction scores array is empty!")
            try { findViewById<TextView>(R.id.prediction).text = "Prediction error: No scores" } catch (e: Exception) {}
            return
        }
        if (scores.size != labels.size) {
            Log.e(TAG, "Prediction scores size (${scores.size}) doesn't match labels size (${labels.size})!")
            try { findViewById<TextView>(R.id.prediction).text = "Error: Score/Label mismatch" } catch (e: Exception) {}
            return
        }

        var maxIndex = -1
        var maxScore = -1 // Use Int for comparison as we convert bytes to unsigned ints (0-255)

        Log.i(TAG, "Raw Prediction Scores (UINT8 Bytes converted to Int 0-255):")
        for (i in scores.indices) {
            // Convert signed Byte (-128 to 127) to unsigned Int (0-255) for correct comparison
            val currentScore = scores[i].toUByte().toInt() // Key conversion!
            val labelName = labels.getOrElse(i) { "Unknown Label $i" } // Safe label access
            Log.i(TAG, "  Index $i: Label=$labelName, Score(0-255)=$currentScore (Raw Byte=${scores[i]})")

            if (currentScore > maxScore) {
                maxScore = currentScore
                maxIndex = i
            }
        }

        val predictedLabel = if (maxIndex != -1) { // Check if maxIndex was updated
            labels[maxIndex] // Access safely as indices were checked
        } else {
            Log.e(TAG, "Prediction failed: No maximum score found (maxIndex remained -1). Scores length: ${scores.size}")
            "Unknown Class"
        }

        // Display just the predicted label name.
        // For UINT8 output, a "percentage" isn't directly meaningful without knowing the scaling/zero-point used during quantization.
        // You could show the raw score (0-255) for debugging if needed.
        val predictionText = predictedLabel
        // val predictionText = "$predictedLabel (Score: $maxScore/255)" // Optional: Show raw score

        try {
            findViewById<TextView>(R.id.prediction).text = predictionText // Ensure R.id.prediction exists
        } catch (e: Exception) {
            Log.e(TAG, "Failed to find prediction TextView (R.id.prediction) to display result", e)
        }

        Log.i(TAG, "--> Predicted Class: $predictedLabel with Raw Score (UINT8 -> Int 0-255): $maxScore")
    }

    // --- Utility to get Bitmap from URI ---
    // Consider adding better error handling or size checks if needed
    fun getBitmapFromUri(uri: Uri?): Bitmap? {
        if (uri == null || uri == Uri.EMPTY) {
            Log.w(TAG, "getBitmapFromUri called with null or empty URI.")
            return null
        }
        var parcelFileDescriptor: android.os.ParcelFileDescriptor? = null
        return try {
            parcelFileDescriptor = contentResolver.openFileDescriptor(uri, "r")
            if (parcelFileDescriptor == null) {
                Log.e(TAG, "ParcelFileDescriptor is null for URI: $uri")
                return null
            }
            val fileDescriptor: FileDescriptor = parcelFileDescriptor.fileDescriptor

            // Basic decoding, consider using BitmapFactory.Options for sampling large images
            val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)

            if (image == null) {
                Log.e(TAG, "BitmapFactory.decodeFileDescriptor returned null for URI: $uri. Is it a valid image file?")
            } else {
                Log.d(TAG, "Successfully decoded Bitmap from URI: $uri (${image.width}x${image.height})")
            }
            image
        } catch (e: SecurityException) {
            Log.e(TAG, "Security Exception getting bitmap from URI: $uri. Check permissions.", e)
            Toast.makeText(this,"Permission denied for image.", Toast.LENGTH_SHORT).show();
            null
        }
        catch (e: Exception) {
            Log.e(TAG, "Failed to get bitmap from URI: $uri", e)
            null
        } finally {
            try {
                parcelFileDescriptor?.close()
            } catch (e: IOException) { // Catch specific IOException
                Log.e(TAG, "Error closing ParcelFileDescriptor", e)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release the interpreter resources when the activity is destroyed
        interpreter?.close()
        Log.d(TAG, "Interpreter closed.")
    }

} // End of MainActivity