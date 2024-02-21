#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <complex>
#include <vector>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <portaudio.h>
#include <pa_asio.h>
#include <fftw3.h>
//#include <rfftw.h>
#include "sndfile.h"
#include <algorithm>

#define M_PI 3.14159265358979323846264338327950288

static unsigned long playbackIndex = 0; // Keeps track of the current index for audio playback as a global var.

struct Shader {
    GLuint programID;  // Shader program ID
    GLuint vertexShaderID;  // Vertex shader ID
    GLuint fragmentShaderID;  // Fragment shader ID

    // Constructor to compile and link shaders
    Shader(const char* vertexPath, const char* fragmentPath) {
        // Load and compile the vertex shader
        vertexShaderID = CompileShader(vertexPath, GL_VERTEX_SHADER);

        // Load and compile the fragment shader
        fragmentShaderID = CompileShader(fragmentPath, GL_FRAGMENT_SHADER);

        // Link the shaders to create the shader program
        programID = LinkShaders(vertexShaderID, fragmentShaderID);
    }

    // Use this shader
    void Use() const {
        glUseProgram(programID);

        //check
        GLint linkStatus;
        glGetProgramiv(programID, GL_LINK_STATUS, &linkStatus);
        if (linkStatus == GL_FALSE) {
            GLint infoLogLength;
            glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
            std::string infoLog(infoLogLength, '\0');
            glGetProgramInfoLog(programID, infoLogLength, nullptr, &infoLog[0]);
            std::cerr << "Shader program linking error: " << infoLog << std::endl;
        }
    }

    // Utility function to compile a shader from a file
    GLuint CompileShader(const char* filePath, GLenum shaderType) {
        GLuint shaderID = glCreateShader(shaderType);

        std::string shaderCode;
        std::ifstream shaderStream(filePath, std::ios::in);

        if (!shaderStream.is_open()) {
            std::cerr << "Failed to open shader file: " << filePath << std::endl;
            return 0; // Return 0 to indicate an error
        }

        shaderCode.assign(
            std::istreambuf_iterator<char>(shaderStream),
            std::istreambuf_iterator<char>()
        );

        shaderStream.close();

        const char* shaderCodePtr = shaderCode.c_str();
        glShaderSource(shaderID, 1, &shaderCodePtr, nullptr);
        glCompileShader(shaderID);

        // Check for compilation errors
        GLint success;
        glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLint infoLogLength;
            glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
            std::string infoLog(static_cast<size_t>(infoLogLength), '\0');
            glGetShaderInfoLog(shaderID, infoLogLength, nullptr, &infoLog[0]);
            std::cerr << "Shader compilation error: " << infoLog << std::endl;
            return -1;
        }

        return shaderID;
    }


    // Utility function to link shaders into a program
    GLuint LinkShaders(GLuint vertexShader, GLuint fragmentShader) {
        GLuint programID = glCreateProgram();
        glAttachShader(programID, vertexShader);
        glAttachShader(programID, fragmentShader);
        glLinkProgram(programID);

        // Check for linking errors
        GLint success;
        glGetProgramiv(programID, GL_LINK_STATUS, &success);
        if (!success) {
            GLint infoLogLength;
            glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
            std::string infoLog(static_cast<size_t>(infoLogLength), '\0');
            glGetProgramInfoLog(programID, infoLogLength, nullptr, &infoLog[0]);
            std::cerr << "Shader program linking error: " << infoLog << std::endl;
            return 0;
        }
        return programID;
    }

    // Destructor to release shader resources
    ~Shader() {
        glUseProgram(0);
        glDeleteShader(vertexShaderID);
        glDeleteShader(fragmentShaderID);
        glDeleteProgram(programID);
    }
};

struct Mesh {
    std::vector<GLfloat> vertices;
    std::vector<GLuint> indices;
    GLuint VAO = 0, VBO = 0, EBO = 0;
    GLuint vertexCount = 0;
    bool quad = true;

    Mesh() {}

    Mesh(const std::vector<GLfloat>& _vertices, const std::vector<GLuint>& _indices = {}) {
        vertices = _vertices;
        bind();

        if (!_indices.empty()) {
            indices = _indices;
            glGenBuffers(1, &EBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
        }
    }

    ~Mesh() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        if (!indices.empty()) {
            glDeleteBuffers(1, &EBO);
        }
    }

    void bind() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    }

    void Draw() {
        glBindVertexArray(VAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
        if (indices.empty()) {
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }
        else {
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        }
        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    void drawQuad() {

        glBindVertexArray(VAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (void*)0);
        if (indices.empty()) {
            glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexCount);
        }
        else {
            glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, 0);
        }
        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
    }
};

// Struct used to pass all the audio features that the playback needs to reproduce audio.
struct UserData {
    std::vector<float> data; // Here we store the entire audio file data.
    double startTime; // App start time
    const int sampleRate;
    const int numChannels;
    bool needsSync = true; // Flag to indicate whether the playback needs to be synced with visual data

    UserData(double st, const int sr, const int nc) : startTime(st), sampleRate(sr), numChannels(nc) {}

    unsigned long currentIndex() {
        double timeElapsed = abs(glfwGetTime()- startTime);
        needsSync = false;
        return static_cast<unsigned long>(timeElapsed*sampleRate*numChannels);
    }
};

//in case in a future we need double buffering for real-time streaming.
//struct DoubleBuffer {
//    std::vector<float> buffers[2]; // 2 buffers with size bufferSize, one for playback and the other for filling (double buffering).
//    int current = 0; // current = 0 -> playback buffer (playback first)
//                     // current = 1 -> fill buffer (fill second)
//    unsigned int bufferSize;
//    std::vector<float> data; // entire audio data
//    unsigned int dataIndex = 0; // current index in the entire audio
//    unsigned int playbackIndex = 0; // current index in playback buffer.
//    bool needsSwap = false; // Flag to indicate whether we reached the end of buffer.
//
//    DoubleBuffer(unsigned int size) : bufferSize(size) {
//        buffers[0].resize(size);
//        buffers[1].resize(size);
//    }
//
//    void swapBuffers() {
//        current ^= 1; //XOR to switch between 0 and 1.
//    }
//
//    std::vector<float> &getFillBuffer() {
//        return buffers[current^1];
//    }
//
//    std::vector<float>& getPlaybackBuffer() {
//        return buffers[current];
//    }
//
//    // this fills the other buffer. When the end of the buffer is reached, buffers are swapped.
//    void fillBuffer() {
//        std::vector<float>& fillBuffer = getFillBuffer();
//        for (unsigned int i = 0; i < bufferSize; ++i, ++dataIndex) {
//            if (dataIndex < data.size())
//                fillBuffer[i] = data[dataIndex];
//            else {
//                fillBuffer[i] = 0.0f; // Fill the rest with silence if at the end of data
//            }
//        }
//    }
//
//    void init() {
//        std::vector<float>& playbackBuffer = getPlaybackBuffer();
//        for (unsigned int i = 0; i < bufferSize; ++i, ++dataIndex) {
//            playbackBuffer[i] = data[i];
//        }
//        fillBuffer();
//    }
//
//    void checkAndSwap() {
//        if (needsSwap) {
//            swapBuffers();
//            playbackIndex = 0;
//            fillBuffer();
//            needsSwap = false;
//        }
//    }
//};

/*struct Camera {
    // Define the camera position & orientation
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    // Define the camera parameters
    float fov = 45.0f;              // Field of view in degrees
    float aspectRatio = 800.0f / 600.0f; // Aspect ratio of the viewport
    float near = 0.1f;              // Near clipping plane
    float far = 100.0f;             // Far clipping plane

    // Create a perspective projection matrix
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, near, far);
    glm::mat4 viewMatrix = glm::lookAt(position, position + target, up);;
    glm::mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

    Camera(float _fov, float _aspectRatio, float _near, float _far) {
        fov = _fov;
        aspectRatio = _aspectRatio;
        near = _near;
        far = _far;
        updateViewProjectionMatrix();
    }

    Camera(glm::vec3 _position, glm::vec3 _target, glm::vec3 _up) {
        position = _position;
        target = _target;
        up = _up;
        updateViewProjectionMatrix();
    }

    Camera() {
        updateViewProjectionMatrix();
    }

    void updateViewMatrix() {
        viewMatrix = glm::lookAt(position, position + target, up);
    }

    void updateProjectionMatrix() {
        projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, near, far); // radians(fov)
    }

    void updateViewProjectionMatrix() {
        updateViewMatrix();
        updateProjectionMatrix();
        // Multiply the projection matrix by the view matrix to get the view-projection matrix
        viewProjectionMatrix = projectionMatrix * viewMatrix;
    }
    void lookAt(glm::vec3 _position, glm::vec3 _target, glm::vec3 _up) {
        position = _position;
        target = _target;
        up = _up;
        updateViewProjectionMatrix();
    }
};   */

// This class manages the buffers of audio for a possible implementation of real time streaming. Not being used for this purpose for now.
class CircularBuffer {
public:
    CircularBuffer(int size) : size_(size), buffer_(size, 0), head_(0) {}

    void PushSample(float sample) {
        buffer_[head_] = sample;
        head_ = (head_ + 1) % size_; // Wrap around when reaching the end
    }

    void init() {
        for (int i = 0; i < size_; i++) {
            buffer_[i] = 0.0f;
        }
    }

    float GetSample(int index) {
        // Calculate the index in the circular buffer
        int actualIndex = (head_ - index - 1 + size_) % size_;
        return buffer_[actualIndex];
    }

    int Size() const {
        return size_;
    }

    float* getData() {
        return buffer_.data();
    }
    std::vector<float> getVector() {
        return buffer_;
    }

    int size_;
    std::vector<float> buffer_;
    int head_;
};

// Performs the Fast Fourier Transform algorithm to a given audio chunk (and interpolates with the magnitudes from the previous iteration for smooth visuals).
std::vector<float> performFFT(const std::vector<float>& audioChunk, std::vector<float> &prev) {

    int N = audioChunk.size();
    std::vector<float> magnitudes(N / 2 + 1); // here we will store the magnitudes
    // Allocate FFTW input and output arrays
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    double* in = (double*)fftw_malloc(sizeof(double) * N);

    // Apply a Hanning window
    for (int i = 0; i < N; ++i) {
        double windowValue = 0.5 * (1 - cos(2 * M_PI * i / (N - 1)));
        in[i] = static_cast<double>(audioChunk[i] * windowValue);
    }

    // Create a plan and execute FFT
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Extract magnitudes and perform easing with the magnitudes of the previous iteration.
    for (int i = 0; i < N/2 + 1; i++) {
        float v = std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]) / N;
        //smoothing over time -->  v = tau * v_prev + (1 - tau) * v
        float tau = 0.8;
        v = tau * prev[i] + (1 - tau) * v;
        prev[i] = v;
        magnitudes[i] = v;
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return magnitudes;
}

// Converts magnitudes to dB scale and normalizes them from a dB_min to dB_max.
std::vector<float> toDecibelAndNormalize(const std::vector<float> &v) {
    int N = v.size();
    float dB_min = -96.0, dB_max = -30.0;
    std::vector<float> out(N);
    for (int i = 0; i < N; ++i) {
        float dB = 20.0 * log10(v[i]);
        double temp = 1 / (dB_max - dB_min) * (dB - dB_min);
        out[i] = (float)std::max(0.0, std::min(temp, 1.0));
    }
    return out;
}

// Simple low-pass filter implementation (unused for now, too costly)
void lowPassFilter(std::vector<float>& data, int downsamplingFactor) {
    std::vector<float> result;
    for (int i = 0; i < data.size() - downsamplingFactor; i++) {
        float sum = 0.0;
        for (int j = 0; j < downsamplingFactor; j++) {
            sum += data[i + j];
        }
        data[i] = sum / downsamplingFactor;
    }
}

// Updates the window of amplitudes. This basically "zooms in & out" the waveform for visualization.
void update_window(const int bufferSize, const int sampleRate, float windowDur, int currentIdx, const std::vector<float>& audioData, std::vector<float>& amplitudeBuffer) {
    // Ensure the window size is not smaller than the buffer's time capacity
    float minimumWindowDur = static_cast<float>(bufferSize) / sampleRate;
    if (windowDur < minimumWindowDur)
        windowDur = minimumWindowDur;

    // Calculate the number of samples to be included in the window
    int windowSamples = static_cast<int>(sampleRate * windowDur);

    // Determine start and end indices for the window
    int startIdx = std::max(currentIdx - windowSamples / 2, 0);
    int endIdx = std::min(startIdx + windowSamples, static_cast<int>(audioData.size()));
    int downsamplingFactor = (endIdx - startIdx) / bufferSize;

    // Downsample and fill the windowed amplitude buffer (TO BE ENHANCED, needs low pass filtering to avoid spatial aliasing).
    amplitudeBuffer.resize(bufferSize);
    for (int i = 0, j = startIdx; i < bufferSize && j < endIdx; ++i, j += downsamplingFactor) {
        (j % 2 == 0) ? amplitudeBuffer[i] = 0.5f * (audioData[j] + audioData[j + 1]): amplitudeBuffer[i] = 0.5f * (audioData[j] + audioData[j - 1]);
    }

}

// PortAudio's playback function that sends audio to an output buffer to be reproduced. Runs in a different thread.
static int playbackCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {

    // Cast userData from void to UserData struct, use its vector, which contains the whole audio.
    UserData* ud = static_cast<UserData*>(userData);
    std::vector<float> &audioData = ud->data;
    float* out = static_cast<float*>(outputBuffer);
    // Start with application time to be in sync with visual data
    if(ud->needsSync) playbackIndex += ud->currentIndex(); 
    // Send audio from the current index to the output buffer to be reproduced.
    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        if (playbackIndex < audioData.size()) {
                                     // (Stereo) -> 2 channels
            *out++ = audioData[playbackIndex++]; // Left
            *out++ = audioData[playbackIndex++]; // Right
        }
        else {
            *out++ = 0.0f; // Left silence
            *out++ = 0.0f; // Right silence
            return paComplete; // If end of playback is reached, stop audio. (NEEDS REFINEMENT)
        }
    }

    return paContinue;
}





int main() {
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    int window_width = 1920; //800
    int window_height = 1080; //450
    float aspectRatio = window_width / window_height;
    // Create a GLFW window
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "OpenGL Window", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window" << std::endl;
        return -1;
    }
    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    double startTime = glfwGetTime(); // Get the start time of the application.

    // Load, compile and link shaders
    Shader shader("shaders/quad2.txt", "shaders/pixel3.txt");
    const char* glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    if (glVersion) {
        std::cout << "OpenGL Version: " << glVersion << std::endl;
    }
    else {
        std::cerr << "Failed to retrieve OpenGL version information." << std::endl;
    }

    // Create a quad mesh.
    std::vector<GLfloat> quadVtx = {
       -1.0f,-1.0f,
        1.0f,-1.0f,
        1.0f, 1.0f,
       -1.0f, 1.0f
    };
    std::vector<GLuint> quadIdx = {
        0,1,2,
        0,2,3
    };
    Mesh quad = Mesh(quadVtx, quadIdx);

    //Open and read fie
    const char* inFileName;
    SNDFILE* inFile;
    SF_INFO inFileInfo;
    int fs;
    inFileName = "audio/all all all.wav"; //all all all, lofifi , delilah, theme
    inFile = sf_open(inFileName, SFM_READ, &inFileInfo);
    if (!inFile) {
        std::cerr << "Failed to open audio file: " << inFileName << std::endl;
        return -1;
    }

    // Extract audio features
    const int numFrames = inFileInfo.frames - 1;
    const int numChannels = inFileInfo.channels;
    const int format = inFileInfo.format;
    const int sampleRate = inFileInfo.samplerate;
    // Save memory for audio buffer (whole audio) in samples and frames.
    std::vector<float> samples(numFrames * numChannels);
    std::vector<float> frames(numFrames);
    // Define the size of the buffers and initialize
    const int bufferSize = 2048;
    CircularBuffer center(bufferSize);     //CircularBuffer left(bufferSize);     //CircularBuffer right(bufferSize);
    std::vector<float> previous(bufferSize * 0.5 + 1); // this buffer stores the fft values of the previous iteration for visual easing of freqs.
    for (int i = 0; i < bufferSize * 0.5 + 1; ++i)
        previous[i] = 0.0; // initialize it to 0s
    std::vector<float> timeWindow; // Buffer of the window of the waveform that will be displayed for visualization.
    float windowDur = 0.0f; //Duration of the window in seconds
    sf_count_t numSamplesRead = sf_readf_float(inFile, samples.data(), numFrames);
    sf_close(inFile);
    if (numSamplesRead < numFrames) {
        std::cerr << "Failed to read the desired number of frames." << std::endl;
        return -1;
    }
    // Initialize the struct userData that will be used for audio playback.
    UserData userData(startTime, sampleRate, numChannels);
    userData.data = samples;

    std::cout << "samples: " << samples.size() << ", frames: " << numFrames << ", sampleRate: " << sampleRate << std::endl;
    std::cout << "max value: " << *std::max_element(samples.begin(), samples.end()) << std::endl;
    std::cout << "min value: " << *std::min_element(samples.begin(), samples.end()) << std::endl;

    // Normalize data for visualization in the range [-1,1]
    float norm_factor = 0.0;
    for (int i = 0; i < samples.size(); i++) {
        norm_factor = std::max(norm_factor, std::abs(samples[i]));
    }
    // Check if norm_factor is not zero to avoid division by zero
    if (norm_factor != 0) {
        // Normalize each sample
        for (int i = 0; i < samples.size(); i++) {
            samples[i] /= norm_factor;
        }
    }
    std::cout << "norm_factor: " << norm_factor << std::endl;

    // Create and configure the data textures
    GLuint dataTexture;
    glGenTextures(1, &dataTexture);
    glBindTexture(GL_TEXTURE_2D, dataTexture);
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    int textureWidth = floor(bufferSize / 2); // Set the width according to your data
    int textureHeight = 2; // 2 channels -> (amplitude, fft)
    // Right after creating the texture and setting its parameters
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth, textureHeight, 0, GL_RGB, GL_FLOAT, NULL);

    // PortAudio (audio playback)
    PaStream* stream;
    PaError err;
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "Failed to initialize PortAudio" << std::endl;
        return -1; // Or handle the error as appropriate
    }
    // Open an audio I/O stream.
    err = Pa_OpenDefaultStream(&stream,
        0, // No input channels
        numChannels, // Stereo output
        paFloat32, // 32-bit floating point output
        sampleRate,
        1024, // Frames per buffer
        playbackCallback,
        &userData); // Passing our UserData struct.
    if (err != paNoError) {
        std::cerr << "Failed to open default stream" << std::endl;
        Pa_Terminate();
        return -1; // Or handle the error as appropriate
    }
    // Start the audio stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream" << std::endl;
        Pa_Terminate();
        return -1; // Or handle the error as appropriate
    }

    float elapsedTime = 0.0f;





    // APP MAIN LOOP
    // =============
    while (!glfwWindowShouldClose(window)) {

        glfwGetFramebufferSize(window, &window_width, &window_height);
        glClear(GL_COLOR_BUFFER_BIT);

        // Calculate the current frame and current sample indexes using the time since the application started.
        unsigned int previousFrame = static_cast<unsigned int>(elapsedTime * sampleRate);
        double currentTime = glfwGetTime();
        elapsedTime = static_cast<float>(currentTime - startTime);
        unsigned int currentFrame = static_cast<unsigned int>(elapsedTime * sampleRate);
        unsigned int currentSample = numChannels * currentFrame;
        //unsigned int difference = currentFrame - previousFrame; // the number of frames between each iteration of the main loop.
        // There are ~800 frames of difference between iterations. The closer power of 2 is 1024, so we use a buffer size of 2048
        //std::cout << "diff: " << difference << std::endl;

        // Update the buffer of audio. We are averaging the amplitudes of all channels to a "center" channel (mono).
        for (int i = 0; i < bufferSize; i++) {
            unsigned int frame = currentFrame + i; // skip the other channel
            unsigned int sample = frame * numChannels;
            // Push current channel to center by averaging with the other channel.
            float newSample = (samples[sample] + samples[sample + 1]) * 1/numChannels; // grab samples from audio buffer data and average to mono.
            center.PushSample(newSample);
        }
        // Update the buffer of the windowed amplitudes (for visualization).
        update_window(bufferSize, sampleRate, windowDur, currentSample, samples, timeWindow);
        // Extract (mono) magnitudes and normalize.
        std::vector<float> fftData = performFFT(center.getVector(), previous);
        fftData = toDecibelAndNormalize(fftData);

        // Define the data you want to store in the texture (audio amplitude L, R and FFT coefficients)
        std::vector<GLfloat> textureData(textureWidth * textureHeight * 3);
        for (int i = 0; i < textureWidth - 1; i++) {
            float amplitude = 0.5 * (timeWindow[2 * i] + timeWindow[2 * i + 1]); // Average L,R to mono channel sample
            textureData[i * 3] = 0.5 * (amplitude+1.0); //Re-normalization from -1,1 to 0,1
            //textureData[i * 3] = 0.5*(center.buffer_[i]+1.0); // R component of first row (AMPLITUDE)
            textureData[i * 3 + 1] = 0.0f;                      // G component of first row (unused)
            textureData[i * 3 + 2] = 0.0f;                      // B component of first row (unused)

            textureData[(textureWidth + i) * 3] = fftData[i];  // R component of second row (MAGNITUDE)
            textureData[(textureWidth + i) * 3 + 1] = 0.0f;    // G component of second row (unused)
            textureData[(textureWidth + i) * 3 + 2] = 0.0f;    // B component of second row (unused)
        }

        // Update the texture with new audio data
        glBindTexture(GL_TEXTURE_2D, dataTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGB, GL_FLOAT, textureData.data());

        // Bind the texture to texture unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, dataTexture);

        // Tell the shader the texture is in texture unit 0
        shader.Use(); // Make sure to use the shader before setting uniforms
        GLint audioTextureLocation = glGetUniformLocation(shader.programID, "u_audioTexture");
        if (audioTextureLocation != -1) {
            glUniform1i(audioTextureLocation, 0); // Texture unit 0
        }

        // Pass time and resolution (window size) as uniforms to the shader
        GLint timeLocation = glGetUniformLocation(shader.programID, "u_time");
        glUniform1f(timeLocation, elapsedTime);
        GLint resolutionLocation = glGetUniformLocation(shader.programID, "u_resolution");
        glUniform2f(resolutionLocation, window_width, window_height);
        //Draw call
        quad.drawQuad();

        // Check for OpenGL errors
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << error << std::endl;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // CLEAN UP
    // ========
    // Stop the PortAudio stream
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    // Terminate PortAudio
    Pa_Terminate();

    // Terminate GLFW
    glfwTerminate();

    return 0;
}

