import kikuchipy as kp
import h5py

dict1 = kp.load('/work3/s203768/EMSoftData/dicts/Dict_Ni_in_use.h5')
dict2 = kp.load('/work3/s203768/EMSoftData/simulations_ebsd/Ni-master-30kV-sig-0-thickness-300.h5')
print(dict1.detector)
# print(dict2.detector)
print(dict1)
# print(dict2)
f = h5py.File('/work3/s203768/Real/TWIP_full_map_01.hdf5', 'r')
patterns = f['Scan 1']['EBSD']['Data']['RawPatterns'][:10,:,:]
print(patterns.shape)
data = kp.signals.EBSD(patterns)
# sims = KikuchiDataset(**sims)  # Create a dataset object
xmap1 = data.dictionary_indexing(dictionary = dict1, keep_n = 1)
# xmap2 = data.dictionary_indexing(dictionary = dict2, keep_n = 1)
data.plot()
rots1 = dict.xmap.rotations[xmap1.simulation_indices[:]]
# rots2 = dict.xmap.rotations[xmap2.simulation_indices[:]]

print(rots1.data)
# print(rots2.data)

# print(dict.detector)