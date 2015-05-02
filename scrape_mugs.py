import urllib
import urllib2
import re

#<span class="field-content"><a href="/mugshots/celebrity/hollywood/aidan-quinn"><h3>Aidan Quinn</h3></a></span>
#<div class="mugshot_container center"><img src="http://www.thesmokinggun.com/sites/default/files/imagecache/670xX/photos/thoward2000mug1.jpg"

#Searches for text matching the regular expression
def search_and_find(regex, html):
    pattern = re.compile(regex)
    names = re.findall(pattern,html)
    return names
	
#The main function
def main():

    browser = urllib.URLopener()
#	The URL for the images to be scraped from
    url = 'http://www.thesmokinggun.com/mugshots/celebrities'
#    Base url for the profile of a particular person
    base = 'http://www.thesmokinggun.com/mugshots/celebrity/'
    
    try:
        resp = browser.open(url)
    except:
        print "failed to open main page"
        return -1
    
    html = resp.read()
#    regex for the links to photos on the main page
    regex = '<span class="field-content"><a href="/mugshots/celebrity/(.+?)"><h3>'
    links = search_and_find(regex, html)

    for link in links:
        name = link.split('/')[-1]
        link = base + link
        try:
            resp = browser.open(link)
        except:
            print link, "failed"
            continue
    
        html = resp.read()
#        regex for url of the mugshot
        regex = '<div class="mugshot_container center"><img src="(.+?)"'
#        obtain url of image
        link = search_and_find(regex, html)[0]
        print link
#        download mugshot
        browser.retrieve(link, 'mugshots/' + name + '.jpg')
	
main()
