import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        _, seq_length, batch_size = y_probs.shape

        assert batch_size == 1

        prev_symbol = None
        for t in range(seq_length):
            probs_t = y_probs[:, t, 0]
            best_idx = np.argmax(probs_t)

            path_prob *= probs_t[best_idx]

            if best_idx != blank:
                symbol = self.symbol_set[best_idx - 1]

                if symbol != prev_symbol:
                    decoded_path.append(symbol)
                
                prev_symbol = symbol
            else:
                prev_symbol = None

        decoded_path = "".join(decoded_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def InitializePaths(self, y_probs0):
        """
        Initialize paths with each of the symbols
        
        Input
        -----

        y_probs0 [np.array, dim=(len(symbols) + 1, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        InitialPathsWithFinalBlank, 
        InitialPathsWithFinalSymbol, 
        InitialBlankPathScore,
        InitialPathScore

        """

        InitialBlankPathScore = {}
        InitialPathsWithFinalBlank = set()

        InitialBlankPathScore[""] = y_probs0[0, 0]
        InitialPathsWithFinalBlank.add("")

        InitialPathScore = {}
        InitialPathsWithFinalSymbol = set()

        for i, c in enumerate(self.symbol_set):
            InitialPathScore[c] = y_probs0[i + 1, 0]
            InitialPathsWithFinalSymbol.add(c)
        
        return (
            InitialPathsWithFinalBlank, 
            InitialPathsWithFinalSymbol,
            InitialBlankPathScore, 
            InitialPathScore
        )


    def ExtendWithBlank(self, PathScore, BlankPathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):

        """
        PathsWithTerminalBlank:  set
        PathsWithTerminalSymbol: set
        y 表示第t个时间步的概率
        """

        # First work on paths with terminal blanks 
        # This represents transitions along horizontal trellis edges for blanks

        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = {}

        for path in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0, 0]
        
        # Then extend paths with terminal symbols by blank
        for path in PathsWithTerminalSymbol:
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y[0, 0]
            else:
                UpdatedPathsWithTerminalBlank.add(path)
                UpdatedBlankPathScore[path] = PathScore[path] * y[0, 0]
        
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore
    
    def ExtendWithSymbol(self, PathScore, BlankPathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = {}

        #  First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for i, c in enumerate(self.symbol_set):
                newPath = path + c
                UpdatedPathsWithTerminalSymbol.add(newPath)
                UpdatedPathScore[newPath] = BlankPathScore[path] * y[i + 1, 0]
        
        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            for i, c in enumerate(self.symbol_set):
                if path[-1] == c:
                    newPath = path
                else:
                    newPath = path + c
                if newPath in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[newPath] += PathScore[path] * y[i + 1, 0]
                else:
                    UpdatedPathsWithTerminalSymbol.add(newPath)
                    UpdatedPathScore[newPath] = PathScore[path] * y[i + 1, 0]
        
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
    
    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}

        scorelist = []
        for symble, score in BlankPathScore.items():
            scorelist.append(score)
        
        for symble, score in PathScore.items():
            scorelist.append(score)

        scorelist = sorted(scorelist, reverse=True)
        if len(scorelist) <= self.beam_width:
            cutoff = scorelist[-1]
        else:
            cutoff = scorelist[self.beam_width - 1]

        PrunedPathsWithTerminalBlank = set()
        for path in PathsWithTerminalBlank:
            if BlankPathScore[path] >= cutoff:
                PrunedPathsWithTerminalBlank.add(path)
                PrunedBlankPathScore[path] = BlankPathScore[path]

        PrunedPathsWithTerminalSymbol = set()
        for path in PathsWithTerminalSymbol:
            if PathScore[path] >= cutoff:
                PrunedPathsWithTerminalSymbol.add(path)
                PrunedPathScore[path] = PathScore[path]
        
        return (
            PrunedPathsWithTerminalBlank,
            PrunedPathsWithTerminalSymbol,
            PrunedBlankPathScore,
            PrunedPathScore
        )
    
    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore

        for path in PathsWithTerminalBlank:
            if path in MergedPaths:
                FinalPathScore[path] += BlankPathScore[path]
            else:
                MergedPaths.add(path)
                FinalPathScore[path] = BlankPathScore[path]

        return MergedPaths, FinalPathScore

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]

        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(y_probs[:, 0, :])
        for t in range(1, T):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathScore, BlankPathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, :])
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathScore, BlankPathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, :])

        
        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        bestPath = max(FinalPathScore, key=FinalPathScore.get)

        return bestPath, FinalPathScore