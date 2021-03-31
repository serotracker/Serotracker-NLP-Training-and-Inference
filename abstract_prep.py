def prepare_abstract(title, abstract):
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