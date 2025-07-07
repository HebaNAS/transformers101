// Enhanced Progress Bar Implementation - using WebSlides native API
(function() {
  // Get DOM elements
  const progressBar = document.querySelector('.progress-bar');
  const progressIndicator = document.querySelector('.progress-indicator');
  const progressSection = document.querySelector('.progress-section');
  
  // Wait for WebSlides to be fully initialized
  document.addEventListener('ws:init', function() {
    // WebSlides is now initialized, we can access its API
    if (!window.ws) {
      console.error('WebSlides not found. Progress bar will not work correctly.');
      return;
    }
    
    // Use WebSlides' own count of slides
    const totalSlides = window.ws.maxSlide_;
    console.log(`Total slides from WebSlides API: ${totalSlides}`);
    
    // Get all actual slides from WebSlides
    const slides = window.ws.slides;
    
    // Update progress bar state using WebSlides data
    function updateProgress(slideIndex) {
      // Ensure slideIndex is within bounds
      if (slideIndex < 0 || slideIndex >= totalSlides) return;
      
      // Update progress indicator width
      const progressPercentage = ((slideIndex + 1) / totalSlides) * 100;
      progressIndicator.style.width = `${progressPercentage}%`;
      
      // Update section title using WebSlides' actual slides
      const currentSlide = slides[slideIndex];
      const sectionTitle = currentSlide.el.getAttribute('data-section-title') || '';
      progressSection.textContent = sectionTitle;
      
      console.log(`Progress updated: Slide ${slideIndex + 1}/${totalSlides}`);
    }
    
    // Listen for WebSlides' native slide change event
    document.addEventListener('ws:slide-change', function(event) {
      // WebSlides provides currentSlide0 which is zero-based index
      const currentSlideIndex = event.detail.currentSlide0;
      updateProgress(currentSlideIndex);
    });
    
    // Initialize with the current slide when loaded
    updateProgress(window.ws.currentSlideI_);
  });
  
  // Backup polling method for static sites - will only run if WebSlides is loaded but events aren't firing
  let lastCurrentSlide = null;
  let checkInterval = setInterval(function() {
    // Check if WebSlides exists
    if (!window.ws || !window.ws.currentSlide_) return;
    
    const currentSlide = window.ws.currentSlide_;
    
    // Only update if the slide has changed
    if (currentSlide && currentSlide !== lastCurrentSlide) {
      lastCurrentSlide = currentSlide;
      const slideIndex = window.ws.currentSlideI_;
      
      // Only update if it's a valid index
      if (slideIndex !== undefined && slideIndex >= 0) {
        const progressPercentage = ((slideIndex + 1) / window.ws.maxSlide_) * 100;
        progressIndicator.style.width = `${progressPercentage}%`;
        
        const sectionTitle = currentSlide.el.getAttribute('data-section-title') || '';
        progressSection.textContent = sectionTitle;
      }
    }
  }, 500);
  
  // Clean up interval when page is unloaded
  window.addEventListener('unload', function() {
    clearInterval(checkInterval);
  });
})();
