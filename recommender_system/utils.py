import time


def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timer_end_message = f"  ## timer ## Finished {func.__name__!r} in {run_time:.4f} seconds"
        print(timer_end_message)
        return value

    return wrapper_timer


def get_nr_of_unique_tracks(signal_planting_strategy):

    n_tracks = 2262292

    if "none" not in signal_planting_strategy:
        parts = signal_planting_strategy.split("_")
        additionaltargetsong_parts = [
            part for part in parts if "additionaltargetsong" in part]
        if additionaltargetsong_parts:  # Check if additionaltargetsong is in the strategy
            # Take the first match
            additionaltargetsong_str = additionaltargetsong_parts[0]
            additionaltargetsong = int(
                additionaltargetsong_str.replace("additionaltargetsong", ""))
        else:
            additionaltargetsong = 0
        n_tracks += 1  # adding 1 tracks for injected song by the collective
        # adding tracks for the additional target songs injected song by the collective
        n_tracks += additionaltargetsong
    # if "free_rid" in signal_planting_strategy:
    #    n_tracks += 1  # adding 1 tracks for injected song by the free rider
    return n_tracks


def get_target_song_info(signal_planting_strategy):
    signal = "spotify:track:AlgoCollectiveBestSong"
    parts = signal_planting_strategy.split("_")
    additionaltargetsong_parts = [
        part for part in parts if "additionaltargetsong" in part]
    if additionaltargetsong_parts:  # Check if additionaltargetsong is in the strategy
        # Take the first match
        additionaltargetsong_str = additionaltargetsong_parts[0]
        additionaltargetsong = int(
            additionaltargetsong_str.replace("additionaltargetsong", ""))
    else:
        additionaltargetsong = 0

    signal_track_ids = []
    track_id_by_track_uri = {}
    for i in range(additionaltargetsong + 1):
        signal_track_ids.append(i)
        # replace the target song name ending with the number i
        signal_str = signal[:-len(str(i))] + str(i)
        track_id_by_track_uri[signal_str] = i
    return signal_track_ids, track_id_by_track_uri, additionaltargetsong


def get_train_fraction(signal_planting_strategy):
    parts = signal_planting_strategy.split("_")
    trainfraction_parts = [part for part in parts if "trainfraction" in part]
    if trainfraction_parts:  # Check if trainfraction is in the strategy
        trainfraction_str = trainfraction_parts[0]  # Take the first match
        known_fraction_of_train_set = float(
            trainfraction_str.replace("trainfraction", ""))
    else:
        known_fraction_of_train_set = 1

    return known_fraction_of_train_set


def get_overshoot_int(signal_planting_strategy):
    print("get overshoot int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    overshoot_parts = [part for part in parts if "overshoot" in part]
    if overshoot_parts:  # Check if trainfraction is in the strategy
        overshoot_str = overshoot_parts[0]  # Take the first match
        overshoot = int(overshoot_str.replace("overshoot", ""))
    else:
        overshoot = 0

    print("returning overshoot:", overshoot)
    return overshoot


def get_plantseed(signal_planting_strategy):
    print("get plantseed int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    plantseed_parts = [part for part in parts if "plantseed" in part]
    if plantseed_parts:  # Check if trainfraction is in the strategy
        plantseed = True
    else:
        plantseed = False

    print("returning plantseed:", plantseed)
    return plantseed


def get_replaceseed(signal_planting_strategy):
    print("get replaceseed int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    replaceseed_parts = [part for part in parts if "replaceseed" in part]
    if replaceseed_parts:  # Check if trainfraction is in the strategy
        replaceseed = True
    else:
        replaceseed = False

    print("returning replaceseed:", replaceseed)
    return replaceseed


def get_insertbefore(signal_planting_strategy):
    print("get insertbefore int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    insertbefore_parts = [part for part in parts if "insertbefore" in part]
    if insertbefore_parts:  # Check if trainfraction is in the strategy
        insertbefore = True
    else:
        insertbefore = False

    print("returning insertbefore:", insertbefore)
    return insertbefore


def get_percentile(signal_planting_strategy):
    print("get percentile int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    percentile_parts = [part for part in parts if "percentile" in part]
    if percentile_parts:  # Check if trainfraction is in the strategy
        percentile_str = percentile_parts[0]  # Take the first match
        percentile = int(percentile_str.replace("percentile", ""))
    else:
        percentile = 0

    print("returning percentile:", percentile)
    return percentile

def get_targetx(signal_planting_strategy):
    print("get targetx int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    targetx_parts = [part for part in parts if "target" in part]
    if targetx_parts:  # Check if trainfraction is in the strategy
        targetx_str = targetx_parts[0]  # Take the first match
        targetx = int(targetx_str.replace("target", ""))
    else:
        targetx = None

    print("returning targetx:", targetx)
    return targetx


def get_duplicateseed(signal_planting_strategy):
    print("get duplicateseed int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    duplicateseed_parts = [part for part in parts if "duplicateseed" in part]
    if duplicateseed_parts:  # Check if trainfraction is in the strategy
        duplicateseed = True
    else:
        duplicateseed = False

    print("returning duplicateseed:", duplicateseed)
    return duplicateseed


def get_promotedifferentsongs(signal_planting_strategy):
    print("get promotedifferentsongs int for strategy:", signal_planting_strategy)
    parts = signal_planting_strategy.split("_")
    promotedifferentsongs_parts = [
        part for part in parts if "promotedifferentsongs" in part]
    if promotedifferentsongs_parts:  # Check if trainfraction is in the strategy
        promotedifferentsongs = True
    else:
        promotedifferentsongs = False

    print("returning promotedifferentsongs:", promotedifferentsongs)
    return promotedifferentsongs


def get_onefamoussong(signal_planting_strategy):
    if "onefamoussong" in signal_planting_strategy:
        return "spotify:track:7yyRTcZmCiyzzJlNzGC9Ol"
    else:
        return None


def get_song_to_promote(signal_planting_strategy):
    if "promoteexisting" in signal_planting_strategy:
        if "0" in signal_planting_strategy:
            # frequency: 10054
            # recos without col act: 87
            return "spotify:track:1G391cbiT3v3Cywg8T7DM1"
        elif "1" in signal_planting_strategy:
            # frequency: 10003
            # recos without col act: 122
            return "spotify:track:4IoYz8XqqdowINzfRrFnhi"
        elif "2" in signal_planting_strategy:
            # frequency: 10003
            # recos without col act: 132
            return "spotify:track:4IoYz8XqqdowINzfRrFnhi"
        elif "3" in signal_planting_strategy:
            # frequency: 9940
            # recos without col act: 142
            return "spotify:track:2IpGdrWvIZipmaxo1YRxw5"
        elif "4" in signal_planting_strategy:
            # frequency: 9982
            # recos without col act: 151
            return "spotify:track:1zWZvrk13cL8Sl3VLeG57F"
        elif "5" in signal_planting_strategy:
            # frequency: 9937
            # recos without col act: 162
            return "spotify:track:55OdqrG8WLmsYyY1jijD9b"
        elif "6" in signal_planting_strategy:
            # frequency: 9976
            # recos without col act: 172
            return "spotify:track:6y6jbcPG4Yn3Du4moXaenr"
        elif "7" in signal_planting_strategy:
            # frequency: 9991
            # recos without col act: 185
            return "spotify:track:4NpDZPwSXmL0cCTaJuVrCw"
        elif "8" in signal_planting_strategy:
            # frequency: 10041
            # recos without col act: 230
            return "spotify:track:1OAiWI2oPmglaOiv9fdioU"
        elif "9" in signal_planting_strategy:
            # frequency: 9943
            # recos without col act: 294
            return "spotify:track:4g3Ax56IslQkI6XVfYKVc5"
    return None


def get_severalfamoussongs(signal_planting_strategy):
    if "severalfamoussongs" in signal_planting_strategy:
        return [
            "spotify:track:7GX5flRQZVHRAGd6B4TmDO",
            "spotify:track:2EEeOnHehOozLq4aS0n6SL",
            "spotify:track:7yyRTcZmCiyzzJlNzGC9Ol",
            "spotify:track:3DXncPQOG4VBw3QHh3S817",
            "spotify:track:5dNfHmqgr128gMY2tc5CeJ",
            "spotify:track:4Km5HrUvYTaSUfiSGPJeQR",
            "spotify:track:0SGkqnVQo9KPytSri1H6cF",
            "spotify:track:5hTpBe8h35rJ67eAWHQsJx",
            "spotify:track:3a1lNhkSLSkpJE4MSHpDu9",
            "spotify:track:152lZdxL1OR0ZMW6KquMif"
        ]
    else:
        return None


disable_progress_bar = False
if disable_progress_bar:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
