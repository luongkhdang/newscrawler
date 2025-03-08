"""
GUI application for the NewsCrawler scraper system.
"""

import os
import sys
import csv
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime
import logging
import queue
import urllib.parse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.scrapers.newspaper_scraper import NewspaperScraper, NewspaperScraperConfig
from src.utils.batch_processor import BatchProcessor, BatchConfig
from src.scrapers.scraper_factory import ScraperFactory
from src.utils.url_classifier import URLClassifier
from src.utils.exceptions import ScraperError
from src.utils.nltk_downloader import download_nltk_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper_gui.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK data
try:
    download_nltk_data()
    logger.info("NLTK data initialized successfully")
except Exception as e:
    logger.error(f"Error initializing NLTK data: {e}")


class LogHandler(logging.Handler):
    """Custom logging handler that redirects logs to a queue."""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        self.log_queue.put(self.format(record))


class ScraperGUI(tk.Tk):
    """GUI application for the NewsCrawler scraper system."""
    
    def __init__(self):
        super().__init__()
        
        self.title("NewsCrawler Scraper")
        self.geometry("1000x800")
        self.minsize(800, 600)
        
        # Set up logging queue
        self.log_queue = queue.Queue()
        self.queue_handler = LogHandler(self.log_queue)
        self.queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.queue_handler)
        
        # Initialize variables
        self.urls = []
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.scraper_thread = None
        self.stop_requested = False
        
        # Create GUI components
        self.create_menu()
        self.create_main_frame()
        
        # Start log queue processing
        self.after(100, self.process_log_queue)
    
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load URLs from CSV", command=self.load_urls_from_csv)
        file_menu.add_command(label="Load URLs from Text File", command=self.load_urls_from_text)
        file_menu.add_command(label="Set Output Directory", command=self.set_output_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    def create_main_frame(self):
        """Create the main application frame."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # URL input section
        url_frame = ttk.LabelFrame(main_frame, text="URLs", padding="10")
        url_frame.pack(fill=tk.BOTH, expand=True)
        
        # URL count and buttons
        url_control_frame = ttk.Frame(url_frame)
        url_control_frame.pack(fill=tk.X)
        
        self.url_count_label = ttk.Label(url_control_frame, text="0 URLs loaded")
        self.url_count_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(url_control_frame, text="Load from CSV", command=self.load_urls_from_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(url_control_frame, text="Load from Text", command=self.load_urls_from_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(url_control_frame, text="Clear URLs", command=self.clear_urls).pack(side=tk.LEFT, padx=5)
        
        # URL list
        url_list_frame = ttk.Frame(url_frame)
        url_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.url_listbox = tk.Listbox(url_list_frame, selectmode=tk.EXTENDED)
        self.url_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        url_scrollbar = ttk.Scrollbar(url_list_frame, orient=tk.VERTICAL, command=self.url_listbox.yview)
        url_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.url_listbox.config(yscrollcommand=url_scrollbar.set)
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=10)
        
        # Create a 2-column grid for configuration options
        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill=tk.X)
        
        # Scraper strategy
        ttk.Label(config_grid, text="Scraper Strategy:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.strategy_var = tk.StringVar(value="auto")
        strategy_combo = ttk.Combobox(config_grid, textvariable=self.strategy_var, state="readonly")
        strategy_combo['values'] = ("auto", "newspaper", "feed", "bs4", "puppeteer")
        strategy_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Number of workers
        ttk.Label(config_grid, text="Number of Workers:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.workers_var = tk.IntVar(value=5)
        workers_spinbox = ttk.Spinbox(config_grid, from_=1, to=20, textvariable=self.workers_var, width=5)
        workers_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Batch size
        ttk.Label(config_grid, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_size_var = tk.IntVar(value=100)
        batch_size_spinbox = ttk.Spinbox(config_grid, from_=1, to=1000, textvariable=self.batch_size_var, width=5)
        batch_size_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Request timeout
        ttk.Label(config_grid, text="Request Timeout (s):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.timeout_var = tk.IntVar(value=30)
        timeout_spinbox = ttk.Spinbox(config_grid, from_=1, to=120, textvariable=self.timeout_var, width=5)
        timeout_spinbox.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Rate limit
        ttk.Label(config_grid, text="Rate Limit (s):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.rate_limit_var = tk.DoubleVar(value=1.0)
        rate_limit_spinbox = ttk.Spinbox(config_grid, from_=0.1, to=10.0, increment=0.1, textvariable=self.rate_limit_var, width=5)
        rate_limit_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Retry count
        ttk.Label(config_grid, text="Retry Count:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.retry_count_var = tk.IntVar(value=3)
        retry_count_spinbox = ttk.Spinbox(config_grid, from_=0, to=10, textvariable=self.retry_count_var, width=5)
        retry_count_spinbox.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Disable image extraction
        self.disable_images_var = tk.BooleanVar(value=True)
        disable_images_check = ttk.Checkbutton(config_grid, text="Disable Image Extraction", variable=self.disable_images_var)
        disable_images_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Output directory
        ttk.Label(config_grid, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir_var = tk.StringVar(value=self.output_dir)
        output_dir_entry = ttk.Entry(config_grid, textvariable=self.output_dir_var, width=30)
        output_dir_entry.grid(row=4, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(config_grid, text="Browse", command=self.set_output_directory).grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Scraping", command=self.start_scraping)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Scraping", command=self.stop_scraping, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Test Selected URL", command=self.test_selected_url).pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=10)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        ttk.Label(stats_grid, text="Total URLs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_urls_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_urls_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Processed:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.processed_urls_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.processed_urls_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Successful:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.successful_urls_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.successful_urls_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Failed:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.failed_urls_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.failed_urls_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Start Time:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.start_time_var = tk.StringVar(value="-")
        ttk.Label(stats_grid, textvariable=self.start_time_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Elapsed Time:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.elapsed_time_var = tk.StringVar(value="-")
        ttk.Label(stats_grid, textvariable=self.elapsed_time_var).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
    
    def process_log_queue(self):
        """Process logs from the queue and display them in the log text widget."""
        try:
            while True:
                log_message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, log_message + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
                self.log_queue.task_done()
        except queue.Empty:
            # Schedule to check again
            self.after(100, self.process_log_queue)
    
    def load_urls_from_csv(self):
        """Load URLs from a CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Try to find the URL column
                url_column = None
                if reader.fieldnames:
                    for field in reader.fieldnames:
                        if field.lower() == 'url':
                            url_column = field
                            break
                
                if not url_column:
                    messagebox.showerror("Error", "Could not find a 'url' column in the CSV file.")
                    return
                
                # Read URLs
                urls = []
                for row in reader:
                    if url_column in row and row[url_column].strip():
                        urls.append(row[url_column].strip())
            
            self.urls = urls
            self.update_url_list()
            logger.info(f"Loaded {len(urls)} URLs from {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load URLs from CSV: {str(e)}")
            logger.error(f"Failed to load URLs from CSV: {str(e)}")
    
    def load_urls_from_text(self):
        """Load URLs from a text file (one URL per line)."""
        file_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            self.urls = urls
            self.update_url_list()
            logger.info(f"Loaded {len(urls)} URLs from {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load URLs from text file: {str(e)}")
            logger.error(f"Failed to load URLs from text file: {str(e)}")
    
    def set_output_directory(self):
        """Set the output directory for scraped articles."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir
        )
        
        if directory:
            self.output_dir = directory
            self.output_dir_var.set(directory)
            logger.info(f"Output directory set to {directory}")
    
    def clear_urls(self):
        """Clear the loaded URLs."""
        self.urls = []
        self.update_url_list()
        logger.info("URLs cleared")
    
    def update_url_list(self):
        """Update the URL listbox and count label."""
        self.url_listbox.delete(0, tk.END)
        for url in self.urls:
            self.url_listbox.insert(tk.END, url)
        
        self.url_count_label.config(text=f"{len(self.urls)} URLs loaded")
        self.total_urls_var.set(str(len(self.urls)))
    
    def test_selected_url(self):
        """Test the selected URL with the configured scraper."""
        selected_indices = self.url_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Info", "Please select a URL to test.")
            return
        
        url = self.url_listbox.get(selected_indices[0])
        
        # Create a thread to test the URL
        threading.Thread(target=self._test_url, args=(url,), daemon=True).start()
    
    def _test_url(self, url):
        """Test a URL with the configured scraper (run in a separate thread)."""
        try:
            self.status_var.set(f"Testing URL: {url}")
            logger.info(f"Testing URL: {url}")
            
            # Create scraper configuration
            config = NewspaperScraperConfig(
                user_agent="NewsCrawler-GUI/1.0",
                browser_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                request_timeout=self.timeout_var.get(),
                minimum_content_length=100,  # Lower threshold for testing
                rate_limits={"default": self.rate_limit_var.get()},
                fetch_images=not self.disable_images_var.get()  # Disable image extraction if checkbox is checked
            )
            
            # Create scraper
            scraper = NewspaperScraper(config.__dict__)
            
            # Check if we can fetch the URL
            if not scraper.can_fetch(url):
                logger.error(f"robots.txt disallows access to {url}")
                messagebox.showerror("Error", f"robots.txt disallows access to {url}")
                self.status_var.set("Ready")
                return
            
            # Check if this might be a homepage rather than an article
            domain = urllib.parse.urlparse(url).netloc
            path = urllib.parse.urlparse(url).path
            if not path or path == "/" or path.lower() == "/index.html":
                # This is likely a homepage, ask if the user wants to extract article links instead
                response = messagebox.askyesno(
                    "Homepage Detected",
                    f"The URL {url} appears to be a homepage rather than an article.\n\n"
                    "Would you like to extract article links from this page instead?"
                )
                
                if response:
                    self._extract_article_links(url, scraper)
                    return
            
            # Scrape the URL
            article = scraper.scrape(url)
            
            if article:
                # Print article information
                logger.info(f"Title: {article.title}")
                logger.info(f"Content length: {len(article.content)} characters")
                logger.info(f"Quality score: {article.quality_score:.2f}")
                logger.info(f"Authors: {', '.join(article.metadata.authors) if article.metadata.authors else 'Unknown'}")
                logger.info(f"Published date: {article.metadata.published_date}")
                logger.info(f"Number of images: {len(article.images)}")
                
                # Save article to file
                output_dir = self.output_dir_var.get()
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                domain = article.metadata.source_domain
                filename = f"{domain}_{timestamp}.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved article to {filepath}")
                
                # Show success message
                messagebox.showinfo("Success", f"Successfully scraped article:\nTitle: {article.title}\nSaved to: {filepath}")
            else:
                logger.error("Failed to scrape article")
                messagebox.showerror("Error", "Failed to scrape article")
        
        except ScraperError as e:
            logger.error(f"Scraper error: {str(e)}")
            messagebox.showerror("Scraper Error", str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")
        
        self.status_var.set("Ready")
    
    def _extract_article_links(self, url, scraper):
        """Extract article links from a homepage."""
        try:
            self.status_var.set(f"Extracting article links from: {url}")
            logger.info(f"Extracting article links from: {url}")
            
            # Use newspaper's built-in functionality to extract links
            import newspaper
            from newspaper import Config as NewspaperConfig
            
            # Create newspaper configuration
            config = NewspaperConfig()
            config.browser_user_agent = scraper.config.browser_user_agent
            config.request_timeout = scraper.config.request_timeout
            config.memoize_articles = False
            
            # Build the site
            site = newspaper.build(url, config=config)
            
            # Get article URLs
            article_urls = [article.url for article in site.articles]
            
            if not article_urls:
                logger.warning(f"No article links found on {url}")
                messagebox.showinfo("No Articles Found", f"No article links were found on {url}")
                self.status_var.set("Ready")
                return
            
            # Limit to a reasonable number for display
            if len(article_urls) > 100:
                article_urls = article_urls[:100]
                logger.info(f"Limited to 100 article links (out of {len(site.articles)})")
            
            # Show dialog to select articles
            self._show_article_selection_dialog(article_urls)
            
        except Exception as e:
            logger.error(f"Error extracting article links: {str(e)}")
            messagebox.showerror("Error", f"Error extracting article links: {str(e)}")
            self.status_var.set("Ready")
    
    def _show_article_selection_dialog(self, article_urls):
        """Show a dialog to select articles to add to the URL list."""
        # Create a new top-level window
        dialog = tk.Toplevel(self)
        dialog.title("Select Articles")
        dialog.geometry("800x600")
        dialog.minsize(600, 400)
        dialog.transient(self)
        dialog.grab_set()
        
        # Create a frame for the dialog content
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label
        ttk.Label(frame, text=f"Found {len(article_urls)} article links. Select the ones you want to add:").pack(anchor=tk.W, pady=5)
        
        # Create a listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)
        
        # Add URLs to the listbox
        for url in article_urls:
            listbox.insert(tk.END, url)
        
        # Select all by default
        listbox.select_set(0, tk.END)
        
        # Create buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def select_all():
            listbox.select_set(0, tk.END)
        
        def deselect_all():
            listbox.selection_clear(0, tk.END)
        
        def add_selected():
            selected_indices = listbox.curselection()
            selected_urls = [listbox.get(i) for i in selected_indices]
            
            if selected_urls:
                self.urls.extend(selected_urls)
                self.update_url_list()
                logger.info(f"Added {len(selected_urls)} article URLs to the list")
                messagebox.showinfo("URLs Added", f"Added {len(selected_urls)} article URLs to the list")
            
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Select All", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Selected", command=add_selected).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=5)
        
        # Set the status
        self.status_var.set("Waiting for article selection...")
    
    def start_scraping(self):
        """Start the scraping process."""
        if not self.urls:
            messagebox.showinfo("Info", "No URLs loaded. Please load URLs first.")
            return
        
        # Disable start button and enable stop button
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Reset progress and statistics
        self.progress_var.set(0.0)
        self.processed_urls_var.set("0")
        self.successful_urls_var.set("0")
        self.failed_urls_var.set("0")
        self.start_time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.elapsed_time_var.set("00:00:00")
        
        # Reset stop flag
        self.stop_requested = False
        
        # Create a thread to run the scraping process
        self.scraper_thread = threading.Thread(target=self._run_scraping, daemon=True)
        self.scraper_thread.start()
        
        # Start updating elapsed time
        self.start_time = datetime.now()
        self._update_elapsed_time()
    
    def _update_elapsed_time(self):
        """Update the elapsed time display."""
        if self.scraper_thread and self.scraper_thread.is_alive():
            elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            self.after(1000, self._update_elapsed_time)
    
    def _run_scraping(self):
        """Run the scraping process (in a separate thread)."""
        try:
            self.status_var.set("Initializing scrapers...")
            logger.info("Starting scraping process")
            
            # Create output directory if it doesn't exist
            output_dir = self.output_dir_var.get()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create scraper configuration
            newspaper_config = NewspaperScraperConfig(
                user_agent="NewsCrawler-GUI/1.0",
                browser_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                request_timeout=self.timeout_var.get(),
                rate_limits={"default": self.rate_limit_var.get()},
                fetch_images=not self.disable_images_var.get()  # Disable image extraction if checkbox is checked
            )
            
            logger.info(f"Image extraction is {'disabled' if self.disable_images_var.get() else 'enabled'}")
            
            # Create scrapers
            newspaper_scraper = NewspaperScraper(newspaper_config.__dict__)
            
            # Create scraper factory
            factory = ScraperFactory()
            factory.register_scraper("newspaper", newspaper_scraper)
            
            # Create URL classifier
            classifier = URLClassifier()
            
            # Create batch processor configuration
            batch_config = BatchConfig(
                max_workers=self.workers_var.get(),
                batch_size=self.batch_size_var.get(),
                timeout=self.timeout_var.get(),
                retry_count=self.retry_count_var.get(),
                output_dir=output_dir
            )
            
            # Create batch processor
            processor = BatchProcessor(factory, batch_config)
            
            # Add URLs to processor
            self.status_var.set("Loading URLs...")
            processor.add_urls(self.urls)
            
            # Process URLs
            self.status_var.set("Processing URLs...")
            total_urls = len(self.urls)
            processed_urls = 0
            successful_urls = 0
            failed_urls = 0
            
            # Define callback function to update progress
            def update_progress(article):
                nonlocal processed_urls, successful_urls
                processed_urls += 1
                successful_urls += 1
                progress = (processed_urls / total_urls) * 100
                
                # Update GUI from the main thread
                self.after(0, lambda: self._update_progress(processed_urls, successful_urls, failed_urls, progress))
            
            # Start processing
            results = processor.process(update_progress)
            
            # Update final statistics
            failed_urls = total_urls - successful_urls
            self._update_progress(processed_urls, successful_urls, failed_urls, 100.0)
            
            # Check if stopped
            if self.stop_requested:
                self.status_var.set("Scraping stopped")
                logger.info("Scraping process stopped by user")
            else:
                self.status_var.set("Scraping completed")
                logger.info("Scraping process completed")
                messagebox.showinfo("Complete", f"Scraping completed.\nProcessed: {processed_urls}\nSuccessful: {successful_urls}\nFailed: {failed_urls}")
        
        except Exception as e:
            logger.error(f"Error in scraping process: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error in scraping process: {str(e)}")
        
        finally:
            # Re-enable start button and disable stop button
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def _update_progress(self, processed, successful, failed, progress):
        """Update the progress display."""
        self.processed_urls_var.set(str(processed))
        self.successful_urls_var.set(str(successful))
        self.failed_urls_var.set(str(failed))
        self.progress_var.set(progress)
    
    def stop_scraping(self):
        """Stop the scraping process."""
        if self.scraper_thread and self.scraper_thread.is_alive():
            self.stop_requested = True
            self.status_var.set("Stopping...")
            logger.info("Stopping scraping process...")
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About NewsCrawler",
            "NewsCrawler Scraper GUI\n\n"
            "A graphical interface for the NewsCrawler scraper system.\n\n"
            "Version: 1.0\n"
            "© 2025 NewsCrawler Team"
        )


if __name__ == "__main__":
    app = ScraperGUI()
    app.mainloop() 