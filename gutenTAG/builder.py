name = input("File Name:")

file = open(name + ".yaml", "w")
file.write(f"timeseries:\n- name: {name}\n")

length = (input("Length:"))
file.write("  length: " + str(length) + "\n")
channels = int(input("Channels:"))
file.write("  channels: " + str(channels) + "\n  base-oscillations:\n")
for i in range(0, channels):
    wave = input("Wave (sine, cosine, square):")
    freq = float(input("Frequency:"))
    amp = float(input("Amplitude:"))
    var = float(input("Variance:"))
    freq_mod = float(input("Frequency Modulation:"))
    file.write(f"  - kind: {wave}\n    frequency: {freq}\n    amplitude: {amp}\n    variance: {var}\n    freq-mod: {freq_mod}\n")

anom = int(input("Anomalies:"))
file.write("  anomalies:\n")
for i in range(0, anom):
    pos = (input("Position (beginning,end,middle):"))
    length = int(input("Length (More than half of the waves frequency):"))
    ampFactor = float(input("Amplitude Factor:"))
    channel = int(input("Channel:"))
    file.write(f"  - position: {pos}\n    length: {length}\n    kinds:\n    - kind: amplitude\n      amplitude_factor: {str(ampFactor)}\n    channel: {channel}\n")
file.close()

