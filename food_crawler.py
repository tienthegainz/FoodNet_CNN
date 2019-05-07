from icrawler.builtin import GoogleImageCrawler

def read_food_list(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    


google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': 'your_image_dir'})
google_crawler.crawl(keyword='cat', max_num=5, file_idx_offset=0)
