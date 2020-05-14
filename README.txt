This is currently going to be a brainstorming document for myself:

I have two options for this: I can make it like transpose (which I think would make the most sense for time
comparisons) or I could do it like the NN assignment and split it into a CPU and GPU version. The latter is
slightly nicer because it would mean everything is more elegantly separated. I think I'm able to do it like
the former and still make it elegant though, so long as I don't just keep everything in one file.

I should probably abstract this nicely into a bunch of classes. There should be a Grid? class that stores
the board and all fundamental operations. We can call an update function to compute the next step. It'll
also store all the parameters for the cellular automaton. I'd like to be able to modify them, if need be.