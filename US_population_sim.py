# Imports

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

class Simulation:

  '''
  Run the simulation
  '''

  def __init__(self, population, length=100):

    '''
    Initialize a simulation of given length
    '''

    # Set attributes:
    self.length = length # Number of generations for which the simulation will run (100 by default)
    self.population = population # Set the population the sim is running on

    # For each year in the simulation, individuals die, reproduce and age
    # Call the functions to facilitate this
    
    for year in tqdm(range(self.length)):
      self.reproduction()
      self.immigration()
      self.mortality()
      self.aging()

  def aging(self):

    '''
    Age each individual in the population by one year each year in the simulation
    '''

    # For each individual in the population, increase their age by 1 each generation
    for individual in (self.population.individuals()):
      individual.age += 1

    # Return the updated individuals vector for the population
    return self.population.individuals()

  def mortality(self):

    '''
    Age based mortality calculated from the life table from the population class, for each year in the population
    '''

    # Retrieves the mortality table for males and females from the population classs
    female_mortality_table = self.population.mortality_table()[0]
    male_mortality_table = self.population.mortality_table()[1]

    # For each individual in the population
    for individual in (self.population.individuals()):
        
        # No individuals will live past the highest age specified in the mortality table (119yrs)
        if individual.age > 120:
          survival = 0

        # If the individual is male, male mortality table is used
        elif individual.sex == "Male":
          # The survival value is 100% minus the mortality probability percentage
          survival = 1 - male_mortality_table[individual.age]

        # If the individual is female, female mortality table is used
        elif individual.sex == "Female":
          # The survival value is 100% minus the mortality probability percentage
          survival = 1 - female_mortality_table[individual.age]

        # Draw a random uniform sample 0-1 using numpy
        random_sample = np.random.uniform()

        # If this random number is larger than the survival percentage, that individual dies
        if random_sample > survival:

          # Remove the individual from the population
          self.population.individuals().remove(individual)

    # Return the updated individuals vector for the population
    return self.population.individuals() 
  
  def reproduction(self):

    '''
    Reproduction for each year in the population
    '''

    # Number of births per in the population this year
    number_of_births = (self.population.birth_rate() * (len(self.population.individuals())))

    reproductive_male_count = 0
    reproductive_female_count = 0

    for individual in self.population.individuals():

      if individual.age > 15 and individual.sex == "Male":
        reproductive_male_count += 1
      elif individual.sex == "Female" and individual.age > 15 and individual.age < 45:
        reproductive_female_count += 1

    for i in range(int(number_of_births)):

      # If there are is more than one individual in the population, there is a potential for reproduction in the population
      if reproductive_male_count >= 1 and reproductive_female_count >= 1:

          # Draw two random indexes from the population individuals vector
          mating_individual_index1 = random.randint(0, (len(self.population.individuals())-1))
          mating_individual_index2 = random.randint(0, (len(self.population.individuals())-1))

          # The individuals at these indexes
          mate1 = self.population.individuals()[mating_individual_index1]
          mate2 = self.population.individuals()[mating_individual_index2]

          # If the randomly sampled individuals in the population are not of reproductive age or are of opposite sex, they cannot mate
          # Randomly sample again until conditions are met for reproduction
          while mate1.age < 15 or mate2.age < 15 or mate1.sex == mate2.sex or (mate1.sex == "Female" and mate1.age >= 45) or (mate2.sex == "Female" and mate2.age >= 45):

            # Draw two random indexes from the population individuals vector
            mating_individual_index1 = random.randint(0, (len(self.population.individuals())-1))
            mating_individual_index2 = random.randint(0, (len(self.population.individuals())-1))

            # The individuals at these indexes
            mate1 = self.population.individuals()[mating_individual_index1]
            mate2 = self.population.individuals()[mating_individual_index2]

          # Call the mating function for the two individuals
          self.population.mate(mate1,mate2)

  def immigration(self):

    '''
    Immigration for each year in the population
    '''

    # Number of immigrants in the population this year
    number_of_immigrants = (self.population.immigration_rate() * (len(self.population.individuals())))

    for i in range(int(number_of_immigrants)):
      # Call the mating function for the two individuals
          self.population.immigrate()

class Population:

  '''
  Class for the whole population. 

  Initialization, adding and removing individuals, mating and immigrating

  Functions for returning a list of all individuals in the population, the birth rate the death rate, the immigration rate

  '''

  def __init__(self, size=0):

    '''
    Initialize a population of given size
    '''

    # Set attributes:
    self.size = size # Population starting size (Zero by default)
    self.population = [] # Initialize vector of individuals in population

    # Runs the number of times from the specified population size
    for individual in range(self.size):

      # Randomly generates a sex for the new individuals
      offspring_sex_value = random.randint(0,1)
      if offspring_sex_value == 0:
        sex = "Male"
      else:
        sex = "Female"

      # Generate a random age for the individual (within reproduction age window to avoid breeding depression for first few years of the sim)
      random_age = random.randint(16,44)

      # Calls add_individual() to add the new individuals to population
      self.add_individual(Individual(random_age,sex,"Natural Born"))

  def add_individual(self, individual):

    '''
    Add a given individual to the population
    '''

    # Add individual to population vector
    self.population.append(individual)

  def individuals(self):

    '''
    Return the list of individuals in the population
    '''
    
    # Returns vector with all individuals in the population
    return self.population

  def mortality_table(self):

    '''
    Using data from Social Security Actuarial Life Table, return the chance of mortality at each age for each sex
    '''

    # Tables from Social Security Actuarial Life Table
    # https://www.ssa.gov/oact/STATS/table4c6.html

    female_mortality_table = {0:0.005046,1:0.000349,2:0.000212,3:0.000166,4:0.000137,5:0.000122,6:0.000111,7:0.000103,8:9.8e-05,9:9.5e-05,10:9.6e-05,11:0.000102,12:0.000116,13:0.000139,14:0.00017,15:0.000204,16:0.00024,17:0.000278,18:0.000319,19:0.00036,20:0.000405,21:0.000451,22:0.000491,23:0.000523,24:0.00055,25:0.000575,26:0.000605,27:0.000642,28:0.000691,29:0.000749,30:0.000811,31:0.000872,32:0.000933,33:0.00099,34:0.001046,35:0.001107,36:0.001172,37:0.001236,38:0.001296,39:0.001356,40:0.001423,41:0.001502,42:0.001596,43:0.001709,44:0.00184,45:0.001988,46:0.002152,47:0.002332,48:0.002528,49:0.002744,50:0.00298,51:0.00324,52:0.003529,53:0.003852,54:0.004207,55:0.00459,56:0.004996,57:0.005425,58:0.005874,59:0.006346,60:0.00688,61:0.007454,62:0.008006,63:0.008515,64:0.009025,65:0.00961,66:0.01032,67:0.011158,68:0.012148,69:0.013301,70:0.014662,71:0.01621,72:0.017892,73:0.019701,74:0.0217,75:0.024064,76:0.026814,77:0.029837,78:0.033132,79:0.03681,80:0.041102,81:0.04608,82:0.051658,83:0.057868,84:0.064829,85:0.07269,86:0.081578,87:0.091587,88:0.102774,89:0.11516,90:0.128749,91:0.143532,92:0.159491,93:0.1766,94:0.194825,95:0.213248,96:0.23157,97:0.249466,98:0.266589,99:0.282585,100:0.29954,101:0.317512,102:0.336563,103:0.356756,104:0.378162,105:0.400852,106:0.424903,107:0.450397,108:0.477421,109:0.506066,110:0.53643,111:0.568616,112:0.602733,113:0.638896,114:0.67723,115:0.717864,116:0.759422,117:0.797393,118:0.837263,119:0.879126,120:1}
    male_mortality_table = {0:0.006081,1:0.000425,2:0.00026,3:0.000194,4:0.000154,5:0.000142,6:0.000135,7:0.000127,8:0.000117,9:0.000104,10:9.7e-05,11:0.000106,12:0.000145,13:0.00022,14:0.000324,15:0.000437,16:0.000552,17:0.000676,18:0.000806,19:0.000939,20:0.001079,21:0.001215,22:0.001327,23:0.001406,24:0.001461,25:0.001508,26:0.001559,27:0.001612,28:0.001671,29:0.001734,30:0.001798,31:0.00186,32:0.001926,33:0.001994,34:0.002067,35:0.002147,36:0.002233,37:0.002318,38:0.002399,39:0.002483,40:0.002581,41:0.002697,42:0.002828,43:0.002976,44:0.003145,45:0.003339,46:0.003566,47:0.003831,48:0.004142,49:0.004498,50:0.004888,51:0.005319,52:0.005808,53:0.00636,54:0.00697,55:0.007627,56:0.00832,57:0.009047,58:0.009803,59:0.010591,60:0.011447,61:0.012352,62:0.013248,63:0.014117,64:0.014995,65:0.015987,66:0.017107,67:0.01828,68:0.0195,69:0.020829,70:0.022364,71:0.024169,72:0.026249,73:0.028642,74:0.03138,75:0.034593,76:0.038235,77:0.042159,78:0.046336,79:0.050917,80:0.056205,81:0.062327,82:0.06919,83:0.076844,84:0.085407,85:0.09501,86:0.10577,87:0.117771,88:0.131063,89:0.145666,90:0.161582,91:0.178797,92:0.197287,93:0.217013,94:0.23793,95:0.258655,96:0.278786,97:0.297897,98:0.315556,99:0.331333,100:0.3479,101:0.365295,102:0.38356,103:0.402738,104:0.422875,105:0.444018,106:0.466219,107:0.48953,108:0.514007,109:0.539707,110:0.566692,111:0.595027,112:0.624778,113:0.656017,114:0.688818,115:0.723259,116:0.759422,117:0.797393,118:0.837263,119:0.879126,120:1}
    
    return female_mortality_table, male_mortality_table

  def birth_rate(self):

    '''
    Using data from global consumer and economic statistics company Statista, return the birth rate per person, per year
    '''

    # Rate from Statista, an international company that providing global data on consumers and economics
    # https://www.statista.com/statistics/195943/birth-rate-in-the-united-states-since-1990/

    # In 2019 there were 11.4 births per 1,000 people
    # 11.4 / 1,000 = 0.0114
    # Birth rate per person per year therefore = 0.0114

    birth_rate = 0.0114

    return birth_rate
  
  def immigration_rate (self):

    '''
    Using data from the Census Bureau and the Department of Homeland Security, calculate and return the immigration rate
    '''

    # The US Census Bureau states in 2019 the US population was 328,239,523
    # https://www.census.gov/newsroom/press-releases/2019/popest-nation.html

    # According to the Department of Homeland Security in 2019 1,031,765 people were granted lawful permanent residence (ie Green Card) in the US
    # https://www.dhs.gov/immigration-statistics/yearbook/2019/table1

    # So 2019 there were 328,239,523 people and 1,031,765 of the people in the census were new imigrants
    # That means the percentage of the population imigrating in is and the rate is:
    # 1,031,765 / 328,239,523 = 0.00314332957 per year

    num_immigrants_per_year = 1031765

    total_population = 328239523
  
    immigration_rate = num_immigrants_per_year/total_population

    return immigration_rate

  def mate(self, individual1, individual2):

    '''
    Takes in two individuals that mate and produce an offspring
    '''
  
    # Randomly generates a sex for the offspring
    offspring_sex_value = random.randint(0,1)
    if offspring_sex_value == 0:
      sex = "Male"
    else:
      sex = "Female"

    # Creates an offspring
    offspring = Individual(0,sex, "Natural Born")

    # Call function to add individual to the population
    self.add_individual(offspring)
  
  def immigrate(self):

    '''
    Generates an immigrant and adds them to the population
    '''
  
    # Randomly generates a sex for the offspring
    offspring_sex_value = random.randint(0,1)
    if offspring_sex_value == 0:
      sex = "Male"
    else:
      sex = "Female"

    # According to the Department of Homeland Security in 2019 of the 1,031,765 people were granted lawful permanent residence 
    # (ie Green Card) in the US:
    #https://www.dhs.gov/immigration-statistics/yearbook/2019/table1

      # 150,042 were under 16 years (14.5423% of the total)
      # 75,033 were 16 to 20 years (7.2723% of the total)
      # And 806,690	were 21 years and over (78.1854% of the total)

      # https://www.dhs.gov/immigration-statistics/yearbook/2019/table8

    # Draw a random uniform sample 0-1 using numpy
    random_sample = np.random.uniform()

    # Use the probabilities above to generate a random age for the immigrant
    if random_sample < 0.1454:
      age = random.randint(1,15)
    elif random_sample >= 0.1454 and random_sample < 0.0727:
      age = random.randint(16,20)
    else:
      age = random.randint(21,75)
    
    # Creates an offspring
    immigrant = Individual(age,sex,"Immigrant")

    # Call function to add individual to the population
    self.add_individual(immigrant)

class Individual:

  '''
  Class for an individual in the population. 

  Initialization, reproduction, and characteristics of each individual

  '''

  def __init__(self, age=0, sex="m", immigrant_status = "nb"):

    '''
    Initialize individual and their traits
    '''

    # Change user input to lowercase
    sex = sex.lower()

    # Handles single character input or full word
    if sex == "m" or sex == "male":
      sex = "Male"
    elif sex == "f" or sex == "female":
      sex = "Female"
    else:
      # Throw an exception if the sex is specified incorrectly when creating object
      raise Exception("Invalid Sex")

    # Change user input to lowercase
    immigrant_status = immigrant_status.lower()

    # Handles single character input or full word
    if immigrant_status == "nb" or immigrant_status == "natural born" or immigrant_status == "n b":
      immigrant_status = "Natural_Born"
    elif immigrant_status == "i" or immigrant_status == "immigrant":
      immigrant_status = "Immigrant"
    else:
      # Throw an exception if the immigrant_status is specified incorrectly when creating object
      raise Exception("Invalid immigrant status")
      
    # Set attributes for that individual
    self.age = age
    self.sex = sex
    self.immigrant_status = immigrant_status

  def age(self):

    '''
    Function for returning the age of the individual
    '''

    # Returns the age of the individual
    return self.age
  
  def sex(self):

    '''
    Function for returning the sex of the individual
    '''

    # Returns the sex of the individual
    return self.sex

  def immigrant_status(self):

    '''
    Function for returning if the individual has immigrated into the population
    '''

    # Returns the immigrant_status of the individual
    return self.immigrant_status
    
    def output_stats(population):

  with open('pop_stats_per_ind.tsv', 'w') as stats:

    line = "Age" + "\t" + "Sex" + "\t" + "Immigrant_status" + "\n"
    stats.write(line)

    # Write out the each individuals age and sex for the total populations
    for individual in pop.individuals():
      line = str(individual.age) + "\t" + str(individual.sex) + "\t" + str(individual.immigrant_status) + "\n"
      stats.write(line)

  with open('pop_stats_females.tsv', 'w') as stats:

    line = "Female_Age\n"
    stats.write(line)
    
    # Write out the age stats for the females
    for individual in pop.individuals():

      if individual.sex == "Female":
        line = str(individual.age) + "\n"
        stats.write(line)

  with open('pop_stats_males.tsv', 'w') as stats:

    line = "Male_Age\n"
    stats.write(line)
    
    # Write out the age stats for the males
    for individual in pop.individuals():

      if individual.sex == "Male":
        line = str(individual.age) + "\n"
        stats.write(line)

  final_pop_size = (len(pop.individuals()))
  APGR = ((((final_pop_size-starting_pop_size)/starting_pop_size)*100)/simulation_length)

  with open('pop_stats_summary.tsv', 'w') as summary:

    final_pop_size_str = ("Final Population Size:\t" + str(final_pop_size) + "\n")
    APGR_str = ("Annual Percentage Growth Rate:\t" + str(APGR) + "%" + "\n")

    # Write out the population size and the annual percentage growth
    summary.write(final_pop_size_str)
    summary.write(APGR_str)

  '''

  ************************************************************************************************************************
  According to the US Census the US population grew 0.1% in 2021
  https://www.census.gov/library/stories/2021/12/us-population-grew-in-2021-slowest-rate-since-founding-of-the-nation.html

  According to worldbank the US population grew 0.4% in 2020
  https://data.worldbank.org/indicator/SP.POP.GROW?locations=US

  (Based on running the simulations several times with different starting population numbers and simulation lengths,
  the annual percentage growth rate tends to fall between 0.4% and 0.1% which seems to be accurate given the 
  numbers from the past two years that we see above.)
  ***********************************************************************************************************************

  '''

def plot_output():

  # Plot total population age distribution
  pop_stats = pd.read_csv("pop_stats_per_ind.tsv", sep="\t")

  # Plot female population age distribution
  female_stats = pd.read_csv("pop_stats_females.tsv", sep="\t")

  # Plot male population age distribution
  male_stats = pd.read_csv("pop_stats_males.tsv", sep="\t")

  # Get frequency per age bins for population pyramid
  bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120]

  freq = male_stats.groupby(pd.cut(male_stats["Male_Age"], bins))
  male_binned_age_count = pd.DataFrame(freq.size().reset_index(name = "Freqency"))

  male_freqs = male_binned_age_count["Freqency"].tolist()

  freq_max = []
  freq_max.append(max(male_freqs))

  male_freqs_negative = []

  for freq in male_freqs:
    freq = 0-freq
    male_freqs_negative.append(freq)

  freq = female_stats.groupby(pd.cut(female_stats["Female_Age"], bins))
  female_binned_age_count = pd.DataFrame(freq.size().reset_index(name = "Freqency"))

  female_freqs = female_binned_age_count["Freqency"].tolist()

  freq_max.append(max(female_freqs))

  maximum_frequency = max(freq_max)

  # Population pyrimid
  
  age_range = ["0 to 4","5 to 9","10 to 14","15 to 19","20 to 24","25 to 29","30 to 34","35 to 39","40 to 44","45 to 49","50 to 54","55 to 59","60 to 64","65 to 69","70 to 74","75 to 79","80 to 84","85 to 89","90 to 94","95 to 99","100 to 104","105 to 109","110 to 120"]
  
  population_age_df = pd.DataFrame({"Age": age_range, "Male": male_freqs_negative, "Female": female_freqs})

  age_range.reverse()

  age_pyramid = sns.barplot(x="Male", y="Age", data=population_age_df, order=age_range, color="#4F81BD")

  age_pyramid = sns.barplot(x="Female", y="Age", data=population_age_df, order=age_range, color="#C0504D")

  age_pyramid.set(xlabel="\n    Male                                  Female\n\nPopulation", 
                  ylabel="Age", title = "Simulation Population Pyramid", 
                  xlim=((0-maximum_frequency),maximum_frequency))
  
  pyramid = age_pyramid.get_figure()    
  pyramid.savefig('sim_age_pyramid.png', dpi=1000, bbox_inches = "tight")
  
starting_pop_size = 40000
simulation_length = 200

# Create a population with a starting size of <starting_pop_size>
pop = Population(starting_pop_size)

# Run a similation on the population for <simulation_length> years
sim = Simulation(pop, simulation_length)

# When the simulation runs a progress bar shows:
# The percentage complete, the year in the simulation, the completed run time and the estimated time before completion

# Output the statistics to files
output_stats(pop)

# Plot
plot_output()
