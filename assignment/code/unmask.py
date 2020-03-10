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


# Find how accurate our unmasking was
# Args:
#   unmasked, a (K x N) matrix of unmasked data
#   truth, a (K x N) matrix representing ground truth
#   normalize, set high if want to only measure '*' accuracy
# Ret:
#   accuracy, percent of matching positions

def compare(unmasked, truth):
    num_diff = np.sum(unmasked == truth)
    total = len(truth) * len(truth[0])
    accuracy = num_diff / total
    return accuracy


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

    '''
    count = 0
    for ch in c:
        if ch == '*':
            count += 1
    print(count)
    '''

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
#   unmasked, a (K x N) matrix of unmasked genotypes
# Ret:
#   haplotypes, a (K x 2N) matrix of the haplotypes which explain the genotypes

def phase(block):
    # First sort the individuals by their heterozygous counts

    i, unmasked = block

    individuals = unmasked.transpose()
    sorted_individuals = sort(individuals)
    # sorted_individuals = [(individuals[i], i) for i in range(len(individuals))]

    '''
    for g in sorted_individuals:
        count = 0
        for g2 in sorted_individuals:
            if areCompatible(g[0], g2[0]):
                count += 1
        print(count)
    '''
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

    '''
    print('first phase:')
    print(phases[0][0][0][:10])
    print(phases[0][0][1][:10])
    print(sorted_individuals[0][0][:10])
    '''
    
    #print('length of H')
    #print(len(H))

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

    min_count = heterozygous_count[0][0]

    return sorted_individuals


# Callback for async function
block_haplotypes = []
def collect_haplotypes(result):
    block_haplotypes.append(result)

def main():
    input_path = '../data/example_data_' + sys.argv[1] + '_masked.txt'
    output_path = '../output/example_data_' + sys.argv[1] + '_unmasked.txt'
    truth_path = '../data/example_data_' + sys.argv[1] + '_sol.txt'

    #input_path = '../data/test_data_masked.txt'

    '''
    # Read input
    masked_data = readInput(input_path)
    #truth = readInput(truth_path)

    # Unmask the data
    start = time.time()
    print('Starting Unmask')
    unmasked_data = unmask(masked_data)
    end = time.time()

    print('Unmasked in ' + str(end - start) + ' seconds')
    '''

    '''
    # Check how far off unmasking is from truth
    diff = compare(unmasked_data, truth)
    print(diff)
    '''

    # Now phase unmasked data

    
    # Change this later but for now just grab input from a file
    unmasked_path = '../output/example_data_' + sys.argv[1] + '_unmasked.txt'
    #unmasked_path = '../output/test_data_unmasked.txt'
    unmasked_data = readInput(unmasked_path)

    # For now, convert to int
    unmasked_data = unmasked_data.astype(int)
    
    # Define a blocksize (hyperparamter which changes)
    BLOCK_SIZE = int(sys.argv[2])

    start = time.time()
    print('Starting Phasing')

    num_blocks = int((len(unmasked_data) / BLOCK_SIZE) + 1)

    blocks = [(i, unmasked_data[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]) for i in range(num_blocks)]

    # Initialize the multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    for block in blocks:
        pool.apply_async(phase, args=(block,), callback=collect_haplotypes)

    #block_haplotypes = [(i, pool.apply_async(phase, args=(blocks[i],))) for i in range(num_blocks)]
    pool.close()
    pool.join()

    block_haplotypes.sort()
    all_haplotypes = [block_haplotype[1] for block_haplotype in block_haplotypes]
    all_haplotypes = np.vstack(all_haplotypes)

    end = time.time()
    print('Phased in ' + str(end - start) + ' seconds')
    


    '''
    # Read truth to check
    truth_data = readInput(truth_path)

    # Only take the first block (10 snps)
    truth_data = truth_data[:10]

    truth_data = truth_data.astype(int)
    countHaplotypes(truth_data)
    '''

    # Write output


    '''
    output_path = '../output/test_data_unmasked.txt'
    writeOutput(output_path, unmasked_data)
    '''

    #output_path = '../output/test_data_sol.txt'
    #writeOutput(output_path, haplotypes)


    output_path = '../output/example_data_' + sys.argv[1] + '_block_' + str(BLOCK_SIZE) + '_sol.txt'
    writeOutput(output_path, all_haplotypes)
    

    print('DONE')


def countHaplotypes(data):
    haps = []
    individuals = data.transpose()

    for h in individuals:
        contains = False
        for hap in haps:
            if np.array_equal(h, hap):
                contains = True

        if not contains:
            haps.append(h)

    print(len(haps))

    exit(0)


if __name__ == "__main__":
    main()