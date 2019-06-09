Another inefficient genetic algorithm for drawing images
===
Here's the genetic algorithm. We're going use the features of evolution to generate an image that resembles the reference image, yet we're constrained to use some fixed number of instruments. 
After about 11 thousand generations and 6 hours (on one processor) the result is something like this:
![](https://i.imgur.com/CEL8lgf.png).
An organism is an image, which contains genes.
In our case, genes are **ellipses**.   
An ellipse consists of other **parameters**: x and y positions, width, height, color. 
The algorithm should use lots of these ellipses to draw an image. So the production of the algorithm is always different for the same input and it does not just copy the image and uses some effects. I assume we can call it an art then.


The features we're going to use: selection, crossover and mutation. All in details.
Higl level description:
```python
for number of epochs
    select organisms
    breed new, possibly mutated, organisms 
    Leave the best in population
select the best image in the last generation
```
Note: the algorithm described here will be a basic version. There could many modifications.
### Quick facts:
**Programming language**: Python  
**Image library**: Pillow  
**Drawing with**: (sometimes translucent) ellispes with white outlines  
**Input image type**: jpg  
**Input image preprocessing**: no preprocessing, the same image
**Output image type**: png (RGBA)  
**Output image colors**: any
**Fitness function**: Mean squared deviation, each pixel computed  
**Library used to calculate fitness**: numpy  

**The method of finding right constants**: my guts  

## Installation and running
```
pip install -r requirements.txt
python genetic_v1.py
```
Default input: `input/reference.jpg`

## Algorithm

### Population
We need to have some number of organisms so they can compete with each other and breed. I keep population size fixed.
Initially, population is generated randomly. For example, the best of the grandparents of all images for some image could be:
![](https://i.imgur.com/IayW8dK.png)



###  Genes generation



When a new gene (ellipse) is created, position and shape is choosen at random.  
The color is more complicated. Color is has 4 values - RGBA. It means it also can be translucent. 
There are 2 ways generate a color: 
* completely randomly
* depending on a color of a given point in the reference image

Now I opted for the latter since it is quicker.
But just picking the same color isn't interesting so let's choose a color which is *about* the same as in the reference image. So the artist improvises or just cannot pick the same color that it sees. Just like a human would..


### Selection
Top N and random M organisms are selected for crossover from the population. We want to the best organisms to breed, but use another heuristic - best images now might not produce better images in the future so we need a bit of randomness in selection too.
```python
def selection(population, matching_pairs_number):
    ...
    top_and_lucky = get_top_and_random_organisms(population, matching_pairs_number)
        i = 0
        while i < matching_pairs_number:
            # CROSSOVER
            pair = top_and_lucky[i], top_and_lucky[i + 1]
            # a pair and a number of children
            children = crossover(pair, 4)
            generation.extend(children)
    		i += 2
    ...
```
### Crossover. For lucky set of selected organisms
Random set of pairs interchange their genes  
Set of K organisms is generated from a pair  
New generation gets random genes from parents  
If parents' genes do not align, every extra gene is also a subject to randomness (whether we choose a gene or leave it behind) 
```python
def crossover(parents, children_count):
	parents_genes = get_parents_genes(parents)
	children = []
	for i in range(children_count):
		child_genes = select_random_genes(parents_genes)
		child = Arrancar(child_genes)
		# MUTATION
		if random.random() < mutation_chance or len(child_genes) < 80:
			child.mutate()
		children.append(child)
	return children
```
Also, crossover is where mutation can happen.
When there are not many genes probability of *some* mutation is 1. I did that just to speed up the process.
### Mutation
Organisms can mutate in 8 ways:
* Add a new gene (ellipse)
* Delete a gene
* Delete a gene and add a new gene
* Mutate a gene
* Gene mutation
* There are 4 color parameters
 * 2 position parameters (x, y)
 * 2 shape parameters (width, height)

```python
def mutate(self):
    ...
    x = random.random()
    if x <= GENE_MUTATION_PROBABILITY:
        # MUTATE EXISTING GENE
        random.choice(self.genes).mutate_ellipse()
    elif x < GENE_ADDITION_PROBABILITY and number_of_genes < MAX_GENES_COUNT:
        self.genes.append(Gene())
    elif x < GENE_DELETION_ADDITION_PROBABILITY:
        if self.genes: self.genes.pop(random.randint(0, number_of_genes - 1))
        self.genes.append(Gene())
    else:
        if self.genes: self.genes.pop(random.randint(0, number_of_genes - 1))
    # why not mutate again?
    if random.random() < 0.1:
        self.mutate()
```
#### A *gene* mutation
```python
def mutate_ellipse(self, choice=random.randint(1, 8)):
    # Mutate different parameter randomly
    if choice == 1:
        self.mutate_color(1)
    elif choice == 2:
        self.mutate_color(2)
    elif choice == 3:
        self.mutate_color(3)
    elif choice == 4:
        self.mutate_color(4)
    elif choice == 5:
        self.mutate_x()
    elif choice == 6:
        self.mutate_y()
    elif choice == 7:
        self.mutate_width()
    elif choice == 8:
        self.mutate_height()
```
As you can see a gene is complex and contains other parameters that can be called *genes* as well.

All mutations shift their values according to a normal distribution, where, the mean is the previous value, or just randomly.
For example, ```x``` position mutation
```python
def mutate_x(self):
    self.x = bound_value(self.x + self.x_change(), MIN_X, MAX_X)
def x_change():
    return random.randint(-X_DELTA, X_DELTA)
```
### Fitness function
Pillow library allows to convert images to numpy arrays. So calculating the mean squared deviation is just done by numpy
```python
def fitness(im1,im2):
	i1 = np.array(im1)
	i2 = np.array(im2)
	return np.mean(np.square(i1 - i2))
```
So the basic algorithm is covered. The main function could look like this:
```python
for i in range(epochs_number):
    epoch += 1
    population = selection(population, selection_range())
```
## Tuning the algorithm.
#### Mutation chance changes
If fitnesses don't *change enough* for a while, there are probably a lot of similar organisms in the population. So the mutation probability is increased, then if the mutation probability is *too big*, the probability is decreases.
What does it mean for a fitness to '*change enough*'? I don't know the right answer. I check if the fitness changes by 0.1 for every 50-100 epochs.
#### Sizes of new generated ellipses change
If fitnesses don't *change enough* for a while, I don't want to the algorithm to try big ellipses, sicnce they're likely not to change fitness because they affect other areas, which are probably gaining some fitness.
### Why does it take so long?
**Algorithm**: A different implementation of crossover, mutation and representation of genes could speed up the process. In the basic version, the algorithm slows down as the number of genes increases. Compare drawing 30 ellipses vs. drawing 500. 
One could 
* Use a fixed set of colors and shapes of ellipses. 
* Use nearly complete state formulation.
* Use the exact colors as in the original one.
The list goes on.

**Constants**: There a lot of parameters (the size of the population, selection range, ect.) which I believe can be optimized.
**Hardware**: Everything was copmuted using one core of my laptop. No multiprocessing.

### "Cheating"
Another version of the algorithm. After some number of epochs, one could save the generated image and next time draw on top of it, getting rid of control on previously generated genes. 

### Other inputs with different algorithms / constants:
Original:  
![](https://i.imgur.com/jmt6Gp8.png)  
If we pick the same colors:  
![](https://i.imgur.com/BuPbF88.png)  
Big circles:  
![](https://i.imgur.com/k4s9hyD.png)  
Squished circles:  
![](https://i.imgur.com/41Avhcu.png)   
After applying "cheating" described above and using smaller circles if the algorithm is stuck:  
![](https://i.imgur.com/zxGQI6R.jpg)  
