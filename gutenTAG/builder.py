#TODO put validation on every input

name = input("File Name:")

file = open(name + ".yaml", "w")
file.write(f"timeseries:\n- name: {name}\n")

length = (input("Length:"))
file.write("  length: " + str(length) + "\n")
channels = int(input("Channels:"))
file.write("  channels: " + str(channels) + "\n  base-oscillations:\n")
for i in range(0, channels):
    print("Channel " + str(i)+ ":")
    wave = input("Wave (sine, cosine, square):")
    freq = float(input("Frequency:"))
    amp = float(input("Amplitude:"))
    var = float(input("Variance:"))
    freq_mod = float(input("Frequency Modulation:"))
    file.write(f"  - kind: {wave}\n    frequency: {freq}\n    amplitude: {amp}\n    variance: {var}\n    freq-mod: {freq_mod}\n")

anom = int(input("Anomalies:"))
file.write("  anomalies:\n")
for i in range(0, anom):
    #These are the general inputs for all anomalies
    anomType = (input("Anomaly Type (amplitude, extremum, frequency):")) 
    pos = (input("Position (beginning,end,middle):"))
    length = int(input("Length (More than half of the waves frequency):"))
    channel = int(input("Channel:"))

    #These are the specific inputs for each anomaly
    if anomType == "amplitude":
        ampFactor = float(input("Amplitude Factor:"))
        file.write(f"  - position: {pos}\n    length: {length}\n    kinds:\n    - kind: amplitude\n      amplitude_factor: {str(ampFactor)}\n    channel: {channel}\n")
    elif anomType == "extremum":
        local = (input("Local (local, global):"))
        minMax = (input("Min or Max (min, max):"))
        minMax = True if minMax == "min" else False
        if local == "local":
            local = True
            contextWindow = int(input("Context Window (How many points used to calc extreme):"))
            file.write(f"  - position: {pos}\n    length: {length}\n    kinds:\n    - kind: extremum\n      local: {local}\n      min: {minMax}\n      context_window: {contextWindow}\n    channel: {channel}\n")
        else:
            local = False
            #global
            file.write(f"  - position: {pos}\n    length: {length}\n    kinds:\n    - kind: extremum\n      local: {local}\n      min: {minMax}\n    channel: {channel}\n")
    elif anomType == "frequency":
        freqFactor = float(input("Frequency Factor:"))
        file.write(f"  - position: {pos}\n    length: {length}\n    kinds:\n    - kind: frequency\n      frequency_factor: {str(freqFactor)}\n    channel: {channel}\n")



file.close()

