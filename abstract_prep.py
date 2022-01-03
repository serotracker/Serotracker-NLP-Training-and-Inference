#preprocessing for abstracts and titles
#combined title with a period then the abstract, and gets rid of new lines, tabs, and funny html characters

def prepare_abstract(title, abstract):
  if len(title) == 0:
    text = abstract
  else:
    if title[-1] != '.':
      title = title + '.'
    if len(abstract) > 0:
      text = title + ' ' + abstract
    else:
      text = title
  
  text = text.replace('<h4>', ' ')
  text = text.replace('</h4>', ' ')
  text = text.replace('\n', ' ')
  text = text.replace('\t', ' ')
  return text