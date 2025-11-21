from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

headers = {'Accept-Language': 'en-US,en;q=0.5'}
url = 'https://www.ebay.com/n/all-categories'
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Extracting all html text within <a> tags inside <li> tags inside <ul> tags with class 'sub-cats' (Subcategory names and links)
category_name = []
category_name_links = []
category_subname = []
category_subname_links = []
category_pictures = []
for wrapper in soup.select('div.cat-wrapper'):
    main_tag = wrapper.select_one('h3.cat-title a.cat-url')
    picture_tag = wrapper.select_one('a.cat-img-wrapper img')

    main_name = main_tag.get_text(strip=True) if main_tag else 'N/A'
    main_link = main_tag['href'] if main_tag else 'N/A'
    picture = picture_tag['src'] if picture_tag else 'N/A'

    for sub_tag in wrapper.select('ul.sub-cats li a'):
        sub_link = sub_tag['href'] if sub_tag else 'N/A'
        text = sub_tag.get_text(strip=True)

        if '-' in text:
            subcategory, _ = text.split('-', 1)
            subcategory = subcategory.strip()
        else:
            subcategory = text.strip()

        # duplicate main category info for each subcategory
        category_name.append(main_name)
        category_name_links.append(main_link)
        category_pictures.append(picture)
        category_subname.append(subcategory)
        category_subname_links.append(sub_link)

print(len(category_name), len(category_name_links), len(category_pictures), len(category_subname), len(category_subname_links))

# Create a DataFrame and saving it to CSV
df = pd.DataFrame({
    'Category': category_name,
    'Category Link': category_name_links,
    'Category Picture': category_pictures,
    'Subcategory': category_subname,
    'Subcategory Link': category_subname_links
})

print(df.head())
df.to_csv('ebay_categories.csv', index=False)


# This is the second source: Rotten Tomatoes
'''
# Extracting movie titles and their ratings from Rotten Tomatoes' homepage
headers = {'Accept-Language': 'en-US,en;q=0.5'}
url = 'https://www.rottentomatoes.com/'
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

heading_categories = []
category_links = []
movie_titles = []
critics_scores = []
audience_scores = []

# Loop through each section (category block)
for section in soup.select("section.dynamic-poster-list"):
    # Extract heading text
    heading_tag = section.select_one("h2.unset rt-text")
    heading = heading_tag.get_text(strip=True) if heading_tag else "Uncategorized"

    # Extract "View all" category link
    link_tag = section.select_one("a.a--short[data-track='showmore']")
    heading_link = link_tag.get("href") if link_tag else "N/A"

    # Extract movies inside this section
    for movie in section.select("tile-dynamic"):
        title_tag = movie.select_one("span.p--small, span.dynamic-text-list__item-title")
        critic_tag = movie.select_one('rt-text[slot="criticsScore"]')
        audience_tag = movie.select_one('rt-text[slot="audienceScore"]')

        title = title_tag.get_text(strip=True) if title_tag else "N/A"
        critic = critic_tag.get_text(strip=True) if critic_tag else "N/A"
        audience = audience_tag.get_text(strip=True) if audience_tag else "N/A"

        if title:
            heading_categories.append(heading)
            category_links.append(heading_link)
            movie_titles.append(title)
            critics_scores.append(critic)
            audience_scores.append(audience)

print(heading_categories, category_links, movie_titles, critics_scores, audience_scores)
print(len(heading_categories), len(category_links), len(movie_titles), len(critics_scores), len(audience_scores))

# Create a DataFrame and saving it to CSV
df = pd.DataFrame({
    'Category': heading_categories,
    'Category Link': category_links,
    'Movie Title': movie_titles,
    'Critics Score': critics_scores,
    'Audience Score': audience_scores
})
print(df.head())
df.to_csv('rotten_tomatoes_dataset.csv', index=False)
'''