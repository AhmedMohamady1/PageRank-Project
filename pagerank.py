import os
import random
import re
import sys

DAMPING = 0.85  # Damping factor for PageRank calculation
SAMPLES = 10000  # Number of samples for sampling PageRank estimation

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus2")
    corpus = crawl("corpus2")  # Parse the corpus directory to get page links
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)  # Estimate PageRank using sampling
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)  # Calculate PageRank by iteration
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
            # Store links in the dictionary, excluding links to self (filename)
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
    Generate a transition model for a page in the corpus.
    """
    model = dict()
    links = corpus[page]
    if not links:
        # If the page has no out-links, produce an equal probability distribution across all pages.
        return {pg: 1/len(corpus) for pg in corpus}
    for pg in corpus:
        model[pg] = (1 - damping_factor) / len(corpus)
    for link in links:
        model[link] += damping_factor / len(links)
    return model

def sample_pagerank(corpus, damping_factor, n):
    """
    Estimate PageRank values by sampling.
    """
    page_rank = {page: 0 for page in corpus}  # Initialize PageRank dictionary
    sample = random.choice(list(corpus.keys()))  # Choose a random page to start sampling
    page_rank[sample] += 1  # Increment the count for the sampled page

    for _ in range(1, n):  # Perform sampling for n iterations
        model = transition_model(corpus, sample, damping_factor)  # Generate transition model for the sample
        sample = random.choices(list(model.keys()), weights=model.values(), k=1)[0]  # Choose next page based on transition model
        page_rank[sample] += 1  # Increment the count for the sampled page

    # Normalize the results by dividing each count by the total number of samples
    for page in page_rank:
        page_rank[page] /= n 
    return page_rank

def iterate_pagerank(corpus, damping_factor):
    """
    Calculate PageRank values by iteration until convergence.
    """
    page_rank = {page: 1/len(corpus) for page in corpus}  # Initialize PageRank dictionary with equal probabilities
    new_rank = page_rank.copy()  # Create a copy of PageRank dictionary for updating

    while True:
        for page in page_rank:
            total = float(0)
            for possible_page in corpus:
                if page in corpus[possible_page]:
                    total += page_rank[possible_page] / len(corpus[possible_page])
                if not corpus[possible_page]:
                    total += page_rank[possible_page] / len(corpus)
            new_rank[page] = (1 - damping_factor) / len(corpus) + damping_factor * total

        # Check for convergence by comparing the new rank with the previous rank
        if all(abs(new_rank[page] - page_rank[page]) < 0.001 for page in page_rank):
            break
        page_rank = new_rank.copy()  # Update the PageRank values for the next iteration

    return new_rank

if __name__ == "__main__":
    main()
