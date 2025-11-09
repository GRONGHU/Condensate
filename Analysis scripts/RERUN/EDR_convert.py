import argparse
import time
import pyedr
import numpy as np
import pickle

def main():
    start_time_main = time.time()
    parser = argparse.ArgumentParser(description='Extract energy matrix from EDR file !')
    
    parser.add_argument('--EDR', type=str, required=True)
    parser.add_argument('--group', type=str, required=True)
    parser.add_argument('--enmat', type=str, required=True)
    args = parser.parse_args()

    print("="*60)    
    print(f"Reading energy data in {args.EDR} ...")
    dic = pyedr.edr_to_dict(args.EDR, verbose=True)
    print("="*60)    
    print(f"Reading energy group in {args.group} ...")
    with open("groups.dat", "r") as f:
        group_data=f.read()
    print("="*60)
    print("Energy groups are:")
    energy_type=group_data.split(" ")

    for name in energy_type:
        print(name)
    print("="*60)
    
    frame_length=len(dic['Time'])
    EM=np.zeros([len(energy_type),len(energy_type),frame_length])
    for i in range(len(energy_type)):
        for j in range(i,len(energy_type)):
            name_Coulsr=f"Coul-SR:{energy_type[i]}-{energy_type[j]}"
            name_Coul14=f"Coul-14:{energy_type[i]}-{energy_type[j]}"
            name_LJsr=f"LJ-SR:{energy_type[i]}-{energy_type[j]}"
            name_LJ14=f"LJ-14:{energy_type[i]}-{energy_type[j]}"
            Etot=dic[name_Coulsr]+dic[name_Coul14]+dic[name_LJsr]+dic[name_LJ14]
            EM[i,j]=Etot
    end = time.time()

    print("Converting complete!")  
    print(f"Saving results in {args.enmat} ...")
    file=open(args.enmat,"wb")
    pickle.dump(EM,file)
    file.close()
    print('Running time: %s Seconds'%(end-start_time_main))
    print('\n')

if __name__ == "__main__":
    main()