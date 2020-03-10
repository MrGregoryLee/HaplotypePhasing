import numpy as np
from scipy import stats
import sys
import time
import multiprocessing as mp

# Notation
# K = num snps
# N = num individuals

# Reads the masked data file and stores it as matrix
# Args:
#   path, a string with the path to where the data is stored
# Ret:
#   data, a (K x N) matrix of masked data
def readInput(path):
    f = open(path, 'r')
    data = np.array([np.array(line.rstrip().split(' ')) for line in f])

    return data


# Writes the unmasked data to a file
# Args:
#   path, a string with the path where data should be stored
#   data, a (K x N) matrix of unmasked data
def writeOutput(path, data):
    f = open(path, 'w')

    for snp in data:
        acc = ''
        for individual in snp:
            acc += str(individual) + ' '
        f.write(acc + '\n')


# Take the masked data and unmask it
# Args:
#   masked, a (K x N) matrix of masked data
# Ret:
#   unmasked, a (K x N) matrix of unmasked data
def unmask(masked):
    for i in range(len(masked)):
        # Get the most common homozygous genotype in this snp position across all individuals
        snp = masked[i]
        homozygous = [int(x) for x in snp if x == '0' or x == '2']
        mode = str(stats.mode(homozygous)[0][0])

        for j in range(len(snp)):
            if masked[i][j] == '*':
                masked[i][j] = mode

    return masked

# Takes a genotype and makes the known haplotype (based on homozygous snps)
# Args:
#   genotype, the genotype to get a haplotype for
# Ret:
#   haplotype, (the known parts of) the haplotype associated with genotype
def fillKnown(genotype):
    haplotype = []

    for snp in genotype:
        if snp == 0:
            haplotype.append(0)
        elif snp == 1:
            haplotype.append('*')
        elif snp == 2:
            haplotype.append(1)

    return haplotype


# Given an individual, find the haplotype which is consistent that is expected to be
# consistent with all other individuals (heuristically, checking homozygous on all others)
# Args:
#   individuals, an array of (genotype, index) pairs
#   first, the genotype to be phased
# Ret:
#   H, an array of haplotypes (expected to only have one element)
def getInitialHaplotypes(individuals, first):
    a = fillKnown(first)

    for individual in individuals:
        for i in range(len(first)):
            if a[i] == '*':
                if individual[0][i] == 0:
                    a[i] = 0
                elif individual[0][i] == 2:
                    a[i] = 1

    H = np.array([np.array(a)])
    return H


# Checks if a haplotype is valid
# Args:
#   haplotype: haplotype to be checked
# Ret:
#   True if valid, False otherwise
def isValidHaplotype(haplotype):
    for h in haplotype:
        if h != 0 and h != 1:
            return False
    return True


# Phase an individual using known haplotypes H, or by adding haplotypes to H
# Args:
#   individuals: array of (genotype, index) pairs
#   individual: the genotype to be phased
#   H: array of known haplotypes
# Ret:
#   h1, h2: a haplotype phase which is consistent with the individual
#   H: the list of all known haplotypes
def getHaplotypes(individuals, individual, H):
    
    # First check if it can be constructed from haplotypes in H
    for h1 in H:
        for h2 in H:
            g = h1 + h2
            if np.array_equal(g, individual):
                return (h1, h2), H

    # If it cannot be constructed from haplotypes in H, try to find a complement for some h
    for h in H:
        complement = individual - h

        if isValidHaplotype(complement):
            H = np.append(H, [complement], axis=0)
            return (h, complement), H
    

    # Finally, if we cannot find a complement, must generate two new haplotypes
    h1 = getInitialHaplotypes(individuals, individual)[0]
    h2 = individual - h1

    H = np.append(H, [h1], axis=0)
    H = np.append(H, [h2], axis=0)

    return (h1, h2), H


# Get the haplotypes for the unmasked genotypes
# Args:
#   block = i: the position of the block, unmasked: a (BLOCK_SIZE x N) matrix of unmasked genotypes
# Ret:
#   i: the position of the block, haplotypes: a (BLOCK_SIZE x 2N) matrix of the haplotypes which explain the genotypes

def phase(block):
    
    i, unmasked = block

    # First sort the individuals by their heterozygous counts
    individuals = unmasked.transpose()
    sorted_individuals = sort(individuals)
    
    # H is the list of known haplotypes
    first = sorted_individuals[0][0]
    H = getInitialHaplotypes(sorted_individuals, first)

    phases = []

    for individual in sorted_individuals:
        haplotypes, H = getHaplotypes(sorted_individuals, individual[0], H)

        if haplotypes == None:
            print("Haplotype returned None")
            exit(1)

        phases.append((haplotypes, individual[1]))

    # Now that we have phases, format them as output
    # Resort back into the original order
    phases = sorted(phases, key=lambda x: x[1])

    output = []

    for phase in phases:
        haplotypes = phase[0]
        output.append(haplotypes[0])
        output.append(haplotypes[1])

    output = np.array(output)
    output = output.transpose()

    return i, output


# Sorts the genotype data in terms of number of heterozygous
# Args:
#   individuals, an (N x K) matrix of individual genotypes
# Ret:
#   sorted_individuals, an (N x K) matrix of sorted individuals where each element is a (genotype, index) pair
def sort(individuals):
    # For each individual, count how many occurrences of 1 there are
    heterozygous_count = [(np.count_nonzero(individuals[i] == 1), i) for i in range(len(individuals))]
    heterozygous_count.sort()
    indices = [x[1] for x in heterozygous_count]

    # Now create a new matrix that is sorted according to their heterozygous count
    sorted_individuals = [(individuals[i], i) for i in indices]

    return sorted_individuals


# Callback for async function
block_haplotypes = []
def collect_haplotypes(result):
    block_haplotypes.append(result)

def main():

    input_path = '../data/test_data_masked.txt'
    output_path = '../output/test_data_sol.txt'

    # Define a blocksize (hyperparameter set to 13 based on experimental testing)
    BLOCK_SIZE = 13

    
    # Read input
    masked_data = readInput(input_path)

    # Unmask the data
    start = time.time()
    print('Starting Unmask')

    unmasked_data = unmask(masked_data).astype(int)
    end = time.time()

    print('Unmasked in ' + str(end - start) + ' seconds')
      

    # Now phase unmasked data
    start = time.time()
    print('Starting Phasing')

    num_blocks = int((len(unmasked_data) / BLOCK_SIZE) + 1)

    blocks = [(i, unmasked_data[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]) for i in range(num_blocks)]

    # Initialize the multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    for block in blocks:
        pool.apply_async(phase, args=(block,), callback=collect_haplotypes)

    # Close the pool and wait until all process complete
    pool.close()
    pool.join()

    # Now reassemble the async output in correct order
    block_haplotypes.sort()
    all_haplotypes = [block_haplotype[1] for block_haplotype in block_haplotypes]
    all_haplotypes = np.vstack(all_haplotypes)

    end = time.time()
    print('Phased in ' + str(end - start) + ' seconds')
    

    # Write output
    writeOutput(output_path, all_haplotypes)

    print('DONE')


if __name__ == "__main__":
    main()