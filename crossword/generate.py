import sys

from crossword import *
from operator import itemgetter

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        
        for var in self.crossword.variables:      
            for word in self.domains[var].copy():
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        intersection = self.crossword.overlaps[x,y]
        num_of_x_letter, num_of_y_letter = intersection
        
        deleted_words = 0
        for word_x in self.domains[x].copy():
            matches = 0 
            for word_y in self.domains[y]:
                if word_x[num_of_x_letter] == word_y[num_of_y_letter]:
                    matches +=1
            if matches == 0:
                self.domains[x].remove(word_x)
                deleted_words +=1
        if deleted_words ==0:
            return False
        return True

        

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.
        
        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        for var in self.crossword.variables:
            for neighbor in self.crossword.neighbors(var):
                while self.revise(var,neighbor):
                    self.revise(var,neighbor)
                    
            if not len(self.domains[var]):
                return False
        return True
        

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == len(self.crossword.variables):
            return True
        return False
        

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        list_of_values =[]
        for key, values in assignment.items():
            for value in values:
                list_of_values.append(value)

        for key, value in assignment.items():
            for neighbor in self.crossword.neighbors(key):
                if neighbor in assignment:
                    x,y = self.crossword.overlaps[key,neighbor]

                    if assignment[key][x] != assignment[neighbor][y]:
                        return False
        return True
                        

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        list_of_words = []
        final_list_of_words = []
        for word in self.domains[var].copy():
            eliminated_words = 0 
            self.domains[var].clear()
            self.domains[var].add(word)
            
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                neigbor_copy = self.domains[neighbor].copy()
                self.revise(neighbor, var)
                eliminated_words = eliminated_words+ abs(int(len(self.domains[neighbor])-int(len(neigbor_copy))))
                self.domains[neighbor] = neigbor_copy

            list_of_words.append((word, eliminated_words))
        sorted_list = sorted(list_of_words, key=itemgetter(1))
            
        for item in sorted_list:
            word, number = item
            final_list_of_words.append(word)

        return final_list_of_words

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        dict1 = {}
        positions = []
        for var in  self.crossword.variables:
            if var not in assignment:
                dict1[var] = len(self.domains[var])
                min_value = float("inf")
                for key, value in dict1.items():
                    if value == min_value:
                        positions.append(key)
                    if value < min_value:
                        min_value = value
                        positions = [] 
                        positions.append(key)
        dict1.clear()
        
        for var in positions:
            number_of_neighbors = len(self.crossword.neighbors(var))
            dict1[var] = number_of_neighbors
        return min(dict1,key=dict1.get)
        

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)

        for word in self.order_domain_values(var,assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = word
        
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
