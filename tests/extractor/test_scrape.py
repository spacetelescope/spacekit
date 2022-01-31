


#from spacekit.extractor.scrape import JsonScraper

# TEST JSON SCRAPER
# 1: MAKE H5
# 2: LOAD H5
    # if output_path is None:
    #     output_path = os.getcwd()
    # os.makedirs(output_path, exist_ok=True)
    # fname = os.path.basename(fname).split(".")[0]
    # if h5 is None:
    #     patterns = json_pattern.split(",")
    #     jsc = JsonScraper(
    #         search_path=input_path,
    #         search_patterns=patterns,
    #         file_basename=fname,
    #         crpt=crpt,
    #         output_path=output_path,
    #     )
    #     jsc.json_harvester()
    #     jsc.h5store()
    # else:
    #     jsc = JsonScraper(h5_file=h5).load_h5_file()

# TEST MASTSCRAPER
#self.df = MastScraper(self.df).scrape_mast()