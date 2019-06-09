from PIL import Image, ImageDraw, ImageFont
import random, os
import numpy as np
import copy


class Color:
	def __init__(self, r, g, b, alpha):
		self.r = r
		self.g = g
		self.b = b
		self.alpha = alpha

	def __str__(self):
		return "({},{},{})".format(self.r, self.g, self.b)


def save_image(organism_image, epoch):
	organism_image.save(os.path.join("results", "{}.png".format(epoch)), 'PNG')


def bound_value(v, min_v, max_v):
	return min(max(min_v, v), max_v)


def add_legend(organism, epoch):
	if organism._image is None:
		organism_image = copy.deepcopy(organism.draw_image())
	else:
		organism_image = copy.deepcopy(organism._image)
	font_size = 15
	font = ImageFont.truetype("arial.ttf", font_size)
	draw = ImageDraw.Draw(organism_image)
	output_text = "epoch= " + str(epoch) + ", fitness = " + str(organism.fitness())
	draw.text((0, 0), output_text, fill=(255, 0, 0), font=font)
	return organism_image


def concatenate_images(im1, im2):
	images = im1, im2
	widths, heights = zip(*(i.size for i in images))

	total_width = sum(widths)
	max_height_im = max(heights)

	new_im = Image.new('RGB', (total_width, max_height_im))

	x_offset = 0
	for im in images:
		new_im.paste(im, (x_offset, 0))
		x_offset += im.size[0]
	return new_im

based_on = None
# based_on = "804.png"

reference_image = Image.open('input/reference.jpg')
# reference_image = Image.open('lion.jpg')
reference_image_size = reference_image.size


POPULATION = 50
# maximum number of pairs, not organisms
SELECTION_MAX = 20
SELECTION_MIN = 10
MAX_CHILDREN = 6

MAX_GENES_COUNT = 400
INITIAL_AMOUNT_OF_GENES = 30

INITIAL_MUTATION_CHANCE = 0.2
mutation_chance = INITIAL_MUTATION_CHANCE
gene_mutation_probability = 0.3

IF_NOT_ADDITION_THEN_DELETION_PROBABILITY = 0.75
MUTATION_CHANCE_INCREASE = 0.01
MUTATION_CHANCE_DECREASE = 0.2

MAX_COLOR_DELTA = 60
COLOR_ST_D = 10
MIN_COLOR_DELTA = 0
# COLOR_MEAN = 40
ALPHA_ST_D = 30

MIN_COLOR = 0
MIN_X = 0
MAX_X = reference_image_size[0]
MIN_Y = 0
MAX_Y = reference_image_size[1]
X_DELTA = int(MAX_X / 5)
MIN_X_DELTA = int(MAX_X / 20)
Y_DELTA = int(MAX_Y / 5)
MIN_Y_DELTA = int(MAX_Y / 20)

# MIN_WIDTH = int(MAX_X / 150)
# INITIAL_MAX_WIDTH = int(MAX_X / 110)
# max_width = INITIAL_MAX_WIDTH
# MIN_HEIGHT = int(MAX_Y / 150)
# INITIAL_MAX_HEIGHT = int(MAX_X / 110)
# max_height = INITIAL_MAX_HEIGHT

MIN_WIDTH = int(MAX_X / 50)
INITIAL_MAX_WIDTH = int(MAX_X / 5)
max_width = INITIAL_MAX_WIDTH
MIN_HEIGHT = int(MAX_Y / 50)
INITIAL_MAX_HEIGHT = int(MAX_X / 5)
max_height = INITIAL_MAX_HEIGHT

# MIN_WIDTH_DELTA = MIN_X_DELTA
WIDTH_DELTA = X_DELTA
# MIN_HEIGHT_DELTA = MIN_Y_DELTA
HEIGHT_DELTA = Y_DELTA

MAX_COLOR = 255

width_mean = int(MAX_X / 30)
WIDTH_ST_D = int(MAX_X / 20)
height_mean = int(MAX_Y / 30)
HEIGHT_ST_D = int(MAX_Y / 20)

INITIAL_ADDITION_PROBABILITY = 0.6
GENE_ADDITION_PROBABILITY = INITIAL_ADDITION_PROBABILITY


def generate_new_genes():
	genes = []
	for i in range(INITIAL_AMOUNT_OF_GENES):
		genes.append(Gene())
	return genes


def generate_color():
	alpha = bound_value(int(random.gauss(MAX_COLOR, ALPHA_ST_D)), MIN_COLOR, MAX_COLOR)
	return Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), alpha)


def generate_color_from_point(point):
	pxl = reference_image.getpixel(point)
	alpha = 255 - random.randint(0, 20)
	alpha = random.randint(230, 255)
	#     alpha = 255
	#     r = pxl[0]
	#     g = pxl[1]
	#     b = pxl[2]
	r = bound_value(pxl[0] + random.randint(-20, 20), 0, 255)
	g = bound_value(pxl[1] + random.randint(-20, 20), 0, 255)
	b = bound_value(pxl[2] + random.randint(-20, 20), 0, 255)
	return Color(r, g, b, alpha)


def fitness(im1, im2):
	i1 = np.array(im1)
	i2 = np.array(im2)
	return np.mean(np.square(i1 - i2))


def selection_range():
#     return random.randint(SELECTION_MIN, SELECTION_MAX)
	return SELECTION_MAX


def number_of_children():
	return MAX_CHILDREN


def get_initial_population(size):
	organisms = []
	for i in range(size):
		organisms.append(Organism())
	return organisms


class Gene:
	def __init__(self, color=None):
		self.x = np.random.randint(MAX_X)
		self.y = np.random.randint(MAX_Y)
		if color is not None:
			self.color = color
		else:
			# self.color = generate_color()
			self.color = generate_color_from_point((self.x, self.y))

		# self.width = bound_value(random.gauss(width_mean, WIDTH_ST_D), MIN_WIDTH, max_width)
		# self.height = bound_value(random.gauss(height_mean, HEIGHT_ST_D), MIN_HEIGHT, max_height)
		self.width = random.randint(MIN_WIDTH, max_width)
		self.height = random.randint(MIN_HEIGHT, max_height)

	def __str__(self):
		return "([{} {}],w = {}, h = {})".format(self.x, self.y, self.color, self.width, self.height)

	def mutate_ellipse(self, choice=random.randint(1, 8)):
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

	@staticmethod
	def color_change():
		return random.randint(MIN_COLOR_DELTA, MAX_COLOR_DELTA)

	def mutate_color(self, param=random.choice([1, 2, 3, 4])):
		if param == 1:
			self.color.r = bound_value(self.color.r + self.color_change(), MIN_COLOR, MAX_COLOR)
		elif param == 2:
			self.color.g = bound_value(self.color.g + self.color_change(), MIN_COLOR, MAX_COLOR)
		elif param == 3:
			self.color.b = bound_value(self.color.b + self.color_change(), MIN_COLOR, MAX_COLOR)
		elif param == 4:
			self.color.alpha = bound_value(self.color.alpha + self.color_change(), MIN_COLOR, MAX_COLOR)

	@staticmethod
	def x_change():
		return random.randint(-X_DELTA, X_DELTA)

	@staticmethod
	def y_change():
		return random.randint(-Y_DELTA, Y_DELTA)

	def mutate_x(self):
		self.x = bound_value(self.x + self.x_change(), MIN_X, MAX_X)

	def mutate_y(self):
		self.y = bound_value(self.y + self.y_change(), MIN_Y, MAX_Y)

	@staticmethod
	def width_change():
		return random.randint(-WIDTH_DELTA, WIDTH_DELTA)

	@staticmethod
	def height_change():
		return random.randint(-HEIGHT_DELTA, HEIGHT_DELTA)

	def mutate_width(self):
		self.width = bound_value(self.width + self.width_change(), MIN_WIDTH, INITIAL_MAX_WIDTH)

	def mutate_height(self):
		self.height = bound_value(self.height + self.height_change(), MIN_HEIGHT, INITIAL_MAX_HEIGHT)


class Organism:
	def __init__(self, genes=None):
		self._fitness = None
		self._image = None
		if genes:
			self.genes = genes
		else:
			self.genes = generate_new_genes()

	def fitness(self):
		if self._fitness:
			return self._fitness
		else:
			self._fitness = fitness(reference_image, self.draw_image(based_on))
			return self._fitness

	def draw_image(self, based_on=None):
		if self._image:
			return self._image
		else:
			if based_on is not None:
				image = Image.open(based_on)
			else:
				image = Image.new('RGB', reference_image_size, 'white')
			canvas = ImageDraw.Draw(image, 'RGBA')

			for g in self.genes:
				color = (g.color.r, g.color.g, g.color.b, g.color.alpha)
				canvas.ellipse([g.x - g.width, g.y - g.height, g.x + g.width, g.y + g.height],
							   fill=color)  # outline='white')
			self._image = image
			return image

	def mutate(self, addition=False, mutation=False):
		self._image = None
		if addition:
			self.genes.append(Gene())
			return

		number_of_genes = len(self.genes)
		x = random.random()
		if x <= gene_mutation_probability:
			# MUTATE EXISTING GENE
			random.choice(self.genes).mutate_ellipse()
		if x < GENE_ADDITION_PROBABILITY and number_of_genes < MAX_GENES_COUNT:
			self.genes.append(Gene())
		elif x < IF_NOT_ADDITION_THEN_DELETION_PROBABILITY:
			if self.genes: self.genes.pop(random.randint(0, number_of_genes - 1))
			self.genes.append(Gene())
		else:
			if self.genes: self.genes.pop(random.randint(0, number_of_genes - 1))
		# why not mutate again?
		if random.random() < 0.1:
			self.mutate()

	def __str__(self):
		genes = ""
		for gene in self.genes:
			genes += gene.__str__() + "\n"
		return genes


def select_random_genes(parent_genes):
	"""there are supposed to be 2 parents"""
	child_genes = []
	for i in range(len(parent_genes[0])):
		child_genes.append(copy.deepcopy(random.choice([parent_genes[0][i], parent_genes[1][i]])))
	for i in range(0, len(parent_genes[1]) - len(parent_genes[0])):
		if bool(random.getrandbits(1)):
			child_genes.append(copy.deepcopy(parent_genes[1][i]))
	return child_genes


def get_parents_genes(parents):
	parents_genes = []
	for i in range(len(parents)):
		parents_genes.append(parents[i].genes)
	return sorted(parents_genes, key=lambda x: len(x))


def crossover(parents, children_count):
	parents_genes = get_parents_genes(list(parents))
	children = []
	for i in range(children_count):
		child_genes = select_random_genes(parents_genes)
		child = Organism(child_genes)
		# MUTATION
		if len(child_genes) < 1200:
			if random.random() < 0.5:
				child.mutate(True)
		if random.random() < mutation_chance:
			child.mutate()
		children.append(child)
	return children


def get_top_and_random_organisms(population, matching_pairs_number):
	random_pairs_number = max(0, int(random.gauss(matching_pairs_number / 10, matching_pairs_number / 5)))
	matching_pairs_number = matching_pairs_number * 2
	random_pairs_number = random_pairs_number * 2
	top = population[:matching_pairs_number - random_pairs_number]
	random.shuffle(top)
	top.extend(random.sample(population[matching_pairs_number - random_pairs_number:matching_pairs_number],
							 random_pairs_number))
	return top


def selection(population, matching_pairs_number):
	generation = []
	top_and_lucky = get_top_and_random_organisms(population, matching_pairs_number)
	i = 0
	while i < matching_pairs_number:
		# CROSSOVER
		pair = top_and_lucky[i], top_and_lucky[i + 1]
		# a pair and a number of children
		children = crossover(pair, number_of_children())
		generation.extend(children)
		i += 2

	generation.extend(population)
	generation.sort(key=lambda x: x.fitness())
	population = generation[:POPULATION]
	del generation
	return population


def change_mutation_probabilities(best, current_best):
	global mutation_chance
	global max_width
	global max_height
	if mutation_chance < 1:
		minimal_change_in_fitness = 0.1/2
		if abs(current_best.fitness() - best.fitness()) < minimal_change_in_fitness:
			print("Change mutation probabilities")
			mutation_chance += MUTATION_CHANCE_INCREASE
			if max_width > MIN_WIDTH:
				max_width = int((MIN_WIDTH + max_width) / 2)
			else:
				max_width = INITIAL_MAX_WIDTH
			if max_height > MIN_HEIGHT:
				max_height = int((MIN_HEIGHT + max_height) / 2)
				print('new max height:', max_height)
			else:
				max_height = INITIAL_MAX_HEIGHT
			print("mutation probability:", mutation_chance)
		elif mutation_chance > 0.5:
			mutation_chance = INITIAL_MUTATION_CHANCE*2

def play(epochs_number):
	population = get_initial_population(POPULATION)
	population.sort(key=lambda x: x.fitness())
	epoch = 0
	best = population[0]
	first_image = add_legend(best, epoch)
	first_image = concatenate_images(first_image, reference_image)
	save_image(first_image, epoch)
	
	epochs = []
	fitness_scores = []
	current_best = population[0]
	
	for i in range(epochs_number):
		epoch += 1
		population = selection(population, selection_range())
	
		if epoch % 50 == 0:
			current_best = population[0]
			image_with_text = add_legend(current_best, epoch)
			im = concatenate_images(image_with_text, reference_image)
			save_image(im, epoch)
			print("number of genes:", len(current_best.genes), "fitness:", current_best.fitness())
			epochs.append(epoch)
			fitness_scores.append(current_best.fitness())
			change_mutation_probabilities(best, current_best)
			best = population[0]
		if epoch % 101 == 0:
			current_best = population[0]
			save_image(current_best.draw_image(), epoch)


epochs_number = 5000
play(epochs_number)