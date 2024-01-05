import os 

path_to_reid_from_pc = '/ABSOLUTE/PATH/TO/REID_FROM_PC' #path outside of docker 
path_to_reid_data = '/ABSOLUTE/PATH/TO/DATA'   #to make symbolic link to data work


memory = '724g'
cpus = 72
gpus = 'all'
port = 14000
image = 'benjamintherien/dev:bevfusion_base-pt3d-pyg-o3d-waymo'
name = 'reid'

command = "docker run -v {}:{} -v {}:{} --memory {} --cpus={} --gpus {} -p {}:{} --name {} --rm -it {}".format(
    path_to_reid_from_pc,path_to_reid_from_pc,path_to_reid_data,path_to_reid_data,memory,cpus,gpus,port,port,name,image
)

print("################################################")
print('[run_docker.py executing] ',command)
print("################################################")

os.system(command)