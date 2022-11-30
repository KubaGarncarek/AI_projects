import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = dict()

    if not page in corpus.keys():
        for page in corpus:
            probability_distribution[page] = 1 / len(corpus)

            return probability_distribution

    for link in corpus.keys():
        probability_distribution[link] = (1-damping_factor) / len(corpus)
        if link in corpus[page]:

            probability_distribution[link] += damping_factor / len(corpus[page])

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    returned_rank = dict()
    first_sample = random.choice(list(corpus.keys()))
    
    probability_distribution = transition_model(corpus, first_sample, damping_factor)

    for _ in range(n-1):

        sample = random.choices(list(corpus.keys()), weights=list(probability_distribution.values()), k=1)[0]
        try:
            returned_rank[sample] += 1/n
        except KeyError:
            returned_rank[sample] = 1/n
        probability_distribution = transition_model(corpus, sample, damping_factor)
        
    return returned_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = dict()
    N = len(corpus)
    
    for page in corpus:
        page_rank[page] = 1 / N

    while True:
        is_accurate = True
        for page in corpus:

            old_rank = page_rank[page]
            i_pages = [i_page for i_page in corpus if page in corpus[i_page]]

            page_rank[page] = (1- damping_factor) / N + damping_factor * sum([(page_rank[i] / len(corpus[i])) for i in i_pages])
            new_rank = page_rank[page]

            if abs(new_rank - old_rank) > 0.001:
                is_accurate = False

        if is_accurate == True:
            break

    return page_rank


if __name__ == "__main__":
    main()
