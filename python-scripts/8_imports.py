
import psutil as ps

# Get CPU Info
cpu_cores = ps.cpu_count(logical=True) 
cpu_usage = ps.cpu_percent(interval=1) 
cpu_freq = ps.cpu_freq() 

print(f"CPU Cores: {cpu_cores}") 
print(f"CPU Usage: {cpu_usage} %") 
print(f"CPU Frequency: {cpu_freq.current} MHz \n")

# get RAM Info
ram = ps.virtual_memory()
print(f"RAM Total: {ram.total / (1024 ** 3):.2f} GB")
print(f"RAM Available: {ram.available / (1024 ** 3):.2f} GB")
print(f"RAM Used: {ram.used / (1024 ** 3):.2f} GB")
print(f"RAM Used Percentage: {ram.percent} % \n")

# Get Disk Info
disk = ps.disk_usage('/')
print(f"Disk Total: {disk.total / (1024 ** 3):.2f} GB")
print(f"Disk Used: {disk.used / (1024 ** 3):.2f} GB")
print(f"Disk Free: {disk.free / (1024 ** 3):.2f} GB")
print(f"Disk Used Percentage: {disk.percent} % \n")

# Get Network Info
net_io = ps.net_io_counters()
print(f"Bytes Sent: {net_io.bytes_sent / (1024 ** 2):.2f} MB")
print(f"Bytes Received: {net_io.bytes_recv / (1024 ** 2):.2f} MB")
print(f"Packets Sent: {net_io.packets_sent}")
print(f"Packets Received: {net_io.packets_recv}")
print(f"Errors In: {net_io.errin}")
print(f"Errors Out: {net_io.errout}")
print(f"Drop In: {net_io.dropin}")
print(f"Drop Out: {net_io.dropout} \n")

# Get Battery Info
battery = ps.sensors_battery()
seconds_left = battery.secsleft

if battery:
    print(f"Battery Percentage: {battery.percent} %")
    print(f"Power Plugged In: {battery.power_plugged}")
    if seconds_left == ps.POWER_TIME_UNLIMITED:
        print("Battery is plugged in (unlimited time).")
    elif seconds_left == ps.POWER_TIME_UNKNOWN:
        print("Battery time unknown.")
    else:
        print(seconds_left)
        print(f"Time Left: {seconds_left / 60:.0f} minutes \n")
else:
    print("\nNo Battery Information Available \n")

# Get System Boot Time
boot_time = ps.boot_time()
import time
print(f"\nSystem Boot Time: {time.ctime(boot_time)}")

