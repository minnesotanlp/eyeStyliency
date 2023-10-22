import csv
import random
import pandas as pd

conditions = ['a', 'b', 'c', 'd', 'e']
stimuli = pd.read_csv('EyetrackingStimuli.csv')
cons = {}
incons = {}
questions = {}
'Wikipedia and Stack Exchange are both online communities in which each user has a reputation score. Users with higher reputations have more authority and privileges (the Wikipedia community mainly focuses on content and edits for Wikipedia pages, whereas the Stack Exchange community focuses on computer programming). '
all_contexts = {
     "PositiveTwitter": ["The following tweets were written by people who were feeling good in some way. The writers were feeling happy, hopeful, or otherwise positive; they tried to express that feeling through their tweets.",
                         "Tweets from people who were feeling happy, hopeful, or otherwise positive."],
     "NegativeTwitter": ["The following tweets were written by people who were feeling bad in some way. The writers were feeling dissatisfied, discouraged, or otherwise negative; they tried to \
express that feeling through their tweets.",
                         "Tweets from people who were feeling dissastisfied, discouraged, or otherwise negative."],
     "PositiveMovie review": ["The following excerpts are from movie critics writing about films they enjoyed. The critics are trying to express their positive feelings, which may range from strong praise to lukewarm approval.",
                              "Movie reviews where the reviewer liked the movie."],
     "NegativeMovie review": ["The following excerpts are from movie critics writing about films they disliked. The critics are trying to express their negative feelings, which may range from strong distate to mild disapproval.",
                              "Move reviews where the reviewer disliked the movie."],
     "PoliteSE": ["Recall that Wikipedia and Stack Exchange both have online message boards where each user has a reputation score. The following posts were all written by users with a low reputation \
score (either on Wikipedia or Stack Exchange). These users are talking to people with higher scores, and therefore want to be professional and polite in order to have a chance at improving their score.",
                  "Wikipedia/Stack Exchange posts written by a user with a low reputation score, talking politely to someone with a higher score."],
     "ImpoliteSE": ["Recall that Wikipedia and Stack Exchange both have online message boards where each user has a reputation score. The following posts \
                    were written by users with a high reputation score (either on Wikipedia or Stack Exchange). These users are speaking to people who are generally 'below' them in the community hierarchy.",
                    "Wikipedia/Stack Exchange forum posts written by a user with a high reputation score, talking to someone 'below' them in the community hierarchy."],
     "PoliteTwitter": ["The following tweets were written by people who are trying to build their professional network or who are performing customer service. These people are trying to sound \
professional and polite.",
                       "Tweets written for professional purposes (either networking or customer service)."],
     "ImpoliteTwitter": ["The following tweets were written by people who were feeling scornful, flippant, or otherwise discourteous towards another person. \
The writers are trying to express this feeling towards that other person in their tweets.",
                         "Rude or impolite tweets."],
     "ContextFree": ["This is a random sample of texts containing all kinds of sentiment.", "Random sample of text containing all kinds of sentiment."]
     }

class Block:
    def __init__(self, pos, pol, source, context, stims, questions, ids):
        self.pos = pos
        self.pol = pol
        self.source = source
        self.stims = stims
        self.status = [False for stim in stims]
        self.questions = questions
        self.ids = ids

    def get_context(self):
         return all_contexts[self.name()][0]

    def get_scontext(self):
         return all_contexts[self.name()][1]
     
    def name(self):
        name = ''
        if self.pol:
            name += 'Polite'
        elif self.pol == False:
            name += 'Impolite'
        if self.pos:
            name += 'Positive'
        elif self.pos == False:
            name += 'Negative'
        name += self.source
        return name

    def use_stim(self, index):
        self.status[index] = True

    def get_stims(self, n=9999):
        result = []
        for i in range(len(self.stims)):
            if not self.status[i]:
                result.append([self.stims[i], self.questions[i], self.ids[i]])
        random.shuffle(result)
        found_stims = [s for s,q,i in result]
        for i in range(len(self.stims)):
            if self.stims[i] in found_stims[:n]:
                self.use_stim(i)
        return [s for s, q, i in result[:n]], [q for s, q, i in result[:n]], [i for s, q, i in result[:n]]

    def reset(self):
        self.status = [False for stim in self.stims]

    def get_opposite(self, other_blocks):
        for block in other_blocks:
            if block.source == self.source:
                if self.pos == None and (not self.pol) == block.pol:
                    return block
                if self.pol == None and (not self.pos) == block.pos:
                    return block
        return None

def get_block(pos, pol, src):
     if pol == True:
          label = "Polite"
     elif pol == False:
          label = "Impolite"
     elif pos == True:
          label = "Positive Sentiment"
     elif pos == False:
          label = "Negative Sentiment"
     stims = stimuli.loc[(stimuli['Label'] == label) & (stimuli['Type'] == src), ['Text', 'Comprehension', 'id']]
     if src == "Tweet":
          src = "Twitter"
     block = Block(pos, pol, src, "", stims['Text'].tolist(), stims['Comprehension'].tolist(), stims['id'].tolist())
     block.context = all_contexts[block.name()]
     return block

neg_tw = get_block(False, None, 'Tweet')
pos_tw = get_block(True, None, 'Tweet')
neg_mr = get_block(False, None, 'Movie review')
pos_mr = get_block(True, None, 'Movie review')
pol_tw = get_block(None, True, 'Tweet')
imp_tw = get_block(None, False, 'Tweet')
pol_se = get_block(None, True, 'SE')
imp_se = get_block(None, False, 'SE')
blocks = [neg_tw, pos_tw, neg_mr, pos_mr, pol_tw, imp_tw, pol_se, imp_se]


# loop through each condition
# loop through each block
# each block needs to take 8 items from the block and 2 items from the opposite block
# mark items as used as we go
# each block take different 2 items from opposite block
item_id = 0
with open('blocked_stimuli.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['trial_id', 'stim_id', '$block', '$cond', 'congruent', '$context', '$text', '$question', '$short_context'])
    writer.writerow(['number', 'number', 'string', 'string', 'boolean', 'string', 'string', 'string', 'string'])
    for i in range(len(conditions)):
        print(conditions[i])
        rows = {}
        for block in blocks:
            # get 2 incon stims
            opposite = block.get_opposite(blocks)
            incons = opposite.stims[i*2:i*2+2] # why i? so that each cond has different incon...
            questions = opposite.questions[i*2:i*2+2]
            ids = opposite.ids[i*2:i*2+2]
            opposite.use_stim(i*2)
            opposite.use_stim(i*2+1)
            rows[block.name()] = []
            for incon, quest, n in zip(incons, questions, ids):
                rows[block.name()].append([str(item_id), n, block.name(), conditions[i], 'false', block.get_context(), incon, quest, block.get_scontext()])
                item_id += 1
        for block in blocks:
            stims, quests, ids = block.get_stims(n = 8)
            for stim, quest, n in zip(stims, quests, ids):
                rows[block.name()].append([str(item_id), n, block.name(), conditions[i], 'true', block.get_context(), stim, quest, block.get_scontext()])
                item_id += 1
        rows['ContextFree'] = []
        for block in blocks:
            # put leftovers into context free block
            stims, quest, ids = block.get_stims(n = 9999)
            for stim, quest, n in zip(stims, quests, ids):
                rows['ContextFree'].append([str(item_id), n, 'ContextFree', conditions[i], 'true', all_contexts['ContextFree'][0], stim, quest, block.get_scontext()])
                item_id += 1
            block.reset()  # reset status for next iteration
        for blockname, items in rows.items():
             # if context-free block, just shuffle and get on with it
             if blockname == 'ContextFree':
                 random.shuffle(items)
                 for item in items:
                      writer.writerow(item)
                 continue
             # randomly select 2 indices for the non-congruent items.
             # conditions: cannot be one of the first 2 items, need at least 2 items between.
             # randomly select first two congruent items
             first_i = random.randint(2, 9)
             second_i = random.randint(2, 9)
             while abs(second_i - first_i) < 3:
                  second_i = random.randint(2, 9)
             cons = [i for i in items if i[4] == 'true']
             incons = [i for i in items if not i[4] == 'true']
             if not len(incons) == 2:
                  print('disaster')
             random.shuffle(cons)
             random.shuffle(incons)
             num_questions = 0
             for i in range(len(cons) + len(incons)):
                  if i == first_i:
                       next_item = incons[0]
                  elif i == second_i:
                       next_item = incons[1]
                  else:
                       next_item = cons.pop()
                       tries = 0
                       while (i + 1 == first_i or i + 1 == second_i) and next_item[7] != 'NO_QUESTION' and tries < 10:
                            cons = [next_item] + cons
                            tries += 1
                            next_item = cons.pop()
                       if (i + 1 == first_i or i + 1 == second_i) and next_item[7] != 'NO_QUESTION':
                            next_item[7] == 'NO_QUESTION'
                            
                       
                  if next_item[7] != 'NO_QUESTION':
                       num_questions += 1
                  if num_questions > 2:
                       next_item[7] == 'NO_QUESTION'
                  writer.writerow(next_item)
