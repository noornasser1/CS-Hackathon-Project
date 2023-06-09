from scipy.io import wavfile
#samplerate, data = wavfile.read('test.wav')

import wave
import numpy as np
import matplotlib.pyplot as plt

# Open the WAV file
file_path = "C:\\Users\\hendb\\Desktop\\WAVV.wav"
audio_file = wave.open(file_path, 'r')

# Get the audio properties
num_frames = audio_file.getnframes()
sample_width = audio_file.getsampwidth()
num_channels = audio_file.getnchannels()
sample_rate = audio_file.getframerate()

# Read the audio data
frames = audio_file.readframes(num_frames)

# Convert the raw data to a numpy array
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
samples = np.frombuffer(frames, dtype=dtype_map[sample_width]) #This is the amplitudes array of the signal

# Reshape the array if stereo audio
if num_channels == 2:
    samples = samples.reshape(-1, 2)

# Calculate the time axis
duration = num_frames / sample_rate
time = np.linspace(0, duration, num=len(samples))

"""
# Plot the original waveform
plt.subplot(2, 2, 1)
plt.plot(time, samples[:, 0])  # Assuming mono audio, use samples[:, 1] for second channel
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Audio Waveform')
"""
# Perform Fourier Transform to get the frequency content
fft = np.fft.fft(samples[:, 0])  # Assuming mono audio, use samples[:, 1] for second channel
freq = np.fft.fftfreq(len(samples), d=1/sample_rate)

#fft is the frequency array of the signal
"""
# Plot the frequency spectrum
plt.subplot(2, 2, 2)
plt.plot(freq, np.abs(fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
"""
# Print the parameters
print("WAV Parameters:")
print(f"File Path: {file_path}")
print(f"Number of Frames: {num_frames}")
print(f"Sample Width: {sample_width} bytes")
print(f"Number of Channels: {num_channels}")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Duration: {duration} seconds")




amplitudes = samples[:, 0]

#generating the noise array

noise=[]
outputs=[]
y=10
ln_x=0
for i in range(len(amplitudes)):
    
    ln_x = np.log(y)
    noise.append(ln_x)
    if(amplitudes[i]>0):
        sum_val=amplitudes[i]-ln_x 
        if(sum_val<0):
            sum_val=0
        outputs.append(sum_val)

    elif(amplitudes[i]<0):
        sum_val=amplitudes[i]+ln_x
        if(sum_val>0):
            sum_val=0
        outputs.append(sum_val)

    else:
        outputs.append(0)
    y+=10


anti_samples = -samples
"""
# Plot the anti-waveform
plt.subplot(2, 2, 3)
plt.plot(time, anti_samples[:, 0])  # Assuming mono audio, use anti_samples[:, 1] for second channel
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Anti-Waveform')

# Perform Fourier Transform to get the frequency content of the anti-waveform
anti_fft = np.fft.fft(anti_samples[:, 0])  # Assuming mono audio, use anti_samples[:, 1] for second channel
anti_freq = np.fft.fftfreq(len(anti_samples), d=1/sample_rate)

# Plot the frequency spectrum of the anti-waveform
plt.subplot(2, 2, 4)
plt.plot(anti_freq, np.abs(anti_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Anti-Waveform')

# Display the plots
plt.tight_layout()
plt.show()
"""


#return values
amplitudes = np.array(amplitudes)
anti_outputs=-(np.array(outputs))



i=0
while (amplitudes[i]==0):
    i+=1
print(i)


frames=[]
anti_frames=[]
j=0
while(j<391000):
    inner_frame= amplitudes[j:j+1000]
    inner_anti = anti_outputs[j:j+1000]
    j+=1000
    frames.append(inner_frame)
    anti_frames.append(inner_anti)


frames=np.array(frames)
anti_frames=np.array(anti_frames)

print(len(frames))
print(len(anti_frames))


# Close the audio file
audio_file.close()



