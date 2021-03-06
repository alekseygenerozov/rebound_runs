#!/usr/bin/env python

import ConfigParser
import argparse
import uuid
import sys

import numpy as np
from collections import OrderedDict
sys.path.append('/home/aleksey/rebound/')
import rebound
import random as rand
from bin_analysis import bin_find_sim

# Density function for semimajor axes (Hayden's implementation)
# def density(min1, max1):
# 	xmin = 1 / max1
# 	xmax = 1 / min1
# 	x = np.linspace(xmin, xmax, num=10000)
# 	f = 1 / x
# 	rand = np.random.choice(f)
# 	return rand

# Aleksey's implementation.
# def density2(min1, max1):
#     r=np.random.random(1)[0]
#     return (1./min1-r*(1./min1-1./max1))**-1.

def density(min1, max1, p):
	r=np.random.random(1)[0]
	if p==1:
		return min1*np.exp(r*np.log(max1/min1))
	else:
		return (r*(max1**(1.-p)-min1**(1.-p))+min1**(1.-p))**(1./(1-p))




def heartbeat(sim):
	print(sim.contents.dt, sim.contents.t)
# sim is a pointer to the simulation object,
# thus use contents to access object data.
# See ctypes documentation for details.
	# print(sim.contents.dt)

def get_tde(sim, reb_coll):
	orbits = sim[0].calculate_orbits(primary=sim[0].particles[0])
	p1,p2 = reb_coll.p1, reb_coll.p2
	idx, idx0 = max(p1, p2), min(p1, p2)
	if idx0==0:
		##idx decremented by 1 because there is no orbit 0
		name=sim[0].simulationarchive_filename
		f=open(name.replace('.bin', '_tde'), 'a+')
		f.write('{0} {1} {2} {3} TDE!\n'.format(sim[0].t, orbits[idx-1].a, orbits[idx-1].e, idx))
		f.close()

	return 0

def main():
	parser=argparse.ArgumentParser(
		description='Set up a rebound run')
	parser.add_argument('--config', nargs=1, default='config',
		help='File containing simulation parameters')
	# parser.add_argument('--keep_bins', action='store_true',
	# 	help="Don't delete bins from simulation")


	args=parser.parse_args()
	config_file=args.config
	# keep_bins=args.keep_bins
	#print keep_bins

	#print config_file
	tag=str(uuid.uuid4())

	##Default stellar parameters 
	config=ConfigParser.SafeConfigParser(defaults={'name': 'archive'.format(tag), 'N':'100', 'e':'0.7',
		'gravity':'basic', 'integrator':'ias15', 'dt':'0', 'buffer':'1.', 'keep_bins':'False', \
		'a_min':'1.', 'a_max':'2.', 'i_max':'5.', 'm':'5e-5', 'rt':'1.0e-4', 'coll':'line', 'pRun':'500', 'pOut':'0.2', 
		'p':'2'}, dict_type=OrderedDict)
	# config.optionxform=str
	config.read(config_file)

	##Name of our put file 
	name=config.get('params', 'name')
	name=name+"_"+tag+".bin"
	##Length of simulation and interval between snapshots
	pRun=config.getfloat('params', 'pRun')
	pOut=config.getfloat('params', 'pOut')
	keep_bins=config.getboolean('params', 'keep_bins')
	rt=config.getfloat('params', 'rt')
	coll=config.get('params', 'coll')
	buff=config.getfloat('params', 'buffer')

	print pRun, pOut, rt, coll
	sections=config.sections()
	##Assume sections are in the same order as in the config file.
	sections=sections[1:]
	sim = rebound.Simulation()
	sim.G = 1.	
	sim.add(m = 1, r=rt) # BH
	sim.gravity=config.get('params', 'gravity')
	sim.integrator=config.get('params', 'integrator')
	dt=config.getfloat('params', 'dt')
	if dt:
		sim.dt=dt
	if sim.gravity=='tree':
		##Fixing box, angle, and boundary parameters in the tree code.
		sim.configure_box(10.)
		sim.boundary='open'
		sim.opening_angle2=1.5

	##Add particles; Can have different sections with different types of particles (e.g. heavy and light)
	##see the example config file in repository. Only require section is params which defines global parameters 
	##for the simulation (pRun and pOut).
	nparts={}
	num={}
	for ss in sections:
		num[ss]=int(config.get(ss, 'N'))
		N=int(buff*num[ss])
		e=config.getfloat(ss, 'e')
		m=config.getfloat(ss, 'm')
		a_min=config.getfloat(ss, 'a_min')
		a_max=config.getfloat(ss, 'a_max')
		p=config.getfloat(ss, 'p')
		i_max=config.getfloat(ss, 'i_max')

		M = np.zeros(N + 1)
		for j in range(0, N + 1):
			M[j] = rand.uniform(0, 2 * np.pi)

		N0=len(sim.particles)
		for l in range(0,N): # Adds stars
			a0=density(a_min, a_max, p)
			sim.add(m = m, a = a0, e = e, inc=np.random.uniform(0, i_max * np.pi / 180.0), Omega = 0, omega = 0, M = M[l], primary=sim.particles[0])
		nparts[ss]=(N0,N0+N-1)
		#print N, m, e, a_min, a_max, i_max
	
	f=open('init_disk', 'w')
	sim.move_to_com()
	for ii in range(len(sim.particles)):
		f.write('{0:.16e} {1:.16e} {2:.16e} {3:.16e} {4:.16e} {5:.16e} {6:.16e}\n'.format(sim.particles[ii].x, sim.particles[ii].y, sim.particles[ii].z,\
			sim.particles[ii].vx, sim.particles[ii].vy, sim.particles[ii].vz, sim.particles[ii].m))
	f.close()

	fen=open(name.replace('.bin', '_en'), 'a')
	fen.write(sim.gravity+'_'+sim.integrator+'_'+'{0}'.format(sim.dt))
	if not keep_bins:
		##Integrate forward a small amount time to initialize accelerations.
		sim.move_to_com()
		sim.integrate(1.0e-15)
		##Look for binaries
		bins=bin_find_sim(sim)
		bins=np.array(bins)
		#print len(bins[:,[1,2]])
		##Delete all the binaries that we found. The identification of binaries depends in part on the tidal field 
		##of the star cluster, and this will change as we delete stars. So we repeat the binary 
		##deletion process several times until there are none left.
		while len(bins>0):
			##Delete in reverse order (else the indices would become messed up)
			to_del=(np.sort(np.unique(bins[:,1]))[::-1]).astype(int)
			##print len(to_del)
			for idx in to_del:
				sim.remove(idx)
			sim.integrate(sim.t+sim.t*1.0e-14)
			bins=bin_find_sim(sim)
			N0=1
			for ss in sections:
				del1=len(np.intersect1d(range(nparts[ss][0],nparts[ss][-1]+1), to_del))
				tot1=nparts[ss][-1]-nparts[ss][0]+1
				nparts[ss]=(N0, N0+tot1-del1-1)
				N0=N0+tot1-del1

	print len(sim.particles)
	for ss in sections[::-1]:
		to_del=range(nparts[ss][0]+num[ss], nparts[ss][-1]+1)[::-1]
		for idx in to_del:
			sim.remove(idx)

	
	ms=np.array([pp.m for pp in sim.particles[1:]])
	print len(ms[ms<=np.median(ms)])
	sim.collision=coll
	sim.collision_resolve=get_tde


	##Set up simulation archive for output
	sim.automateSimulationArchive(name,interval=np.pi*pOut,deletefile=True)
	#sim.heartbeat=heartbeat
	sim.move_to_com()

	en=sim.calculate_energy()
	print rebound.__version__
	sim.integrate(pRun*2*np.pi)
	en2=sim.calculate_energy()
	#print abs(en2-en)/en
	fen.write('_{0:2.3g}'.format(abs(en2-en)/en))
	fen.close()





if __name__ == '__main__':
	main()








