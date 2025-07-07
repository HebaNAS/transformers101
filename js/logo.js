// Enhanced logo switching that works with static sites
(function() {
  // Function to check the current slide and update logo visibility
  function updateLogoVisibility() {
    // Find the current slide - which has the 'current' class
    const currentSlide = document.querySelector('#webslides section.current');
    if (!currentSlide) return;
    
    const lightLogo = document.querySelector('.logo');
    const darkLogo = document.querySelector('.dark-logo');
    
    // Check if the current slide has a light background
    const hasLightBackground = 
      currentSlide.classList.contains('bg-white') || 
      currentSlide.classList.contains('bg-light') || 
      currentSlide.classList.contains('bg-brown') ||
      currentSlide.classList.contains('bg-gradient-v') ||
      currentSlide.classList.contains('bg-gradient-gray') ||
      currentSlide.classList.contains('bg-gradient-white');
    
    // If light background, show dark logo
    if (hasLightBackground) {
      lightLogo.style.opacity = '0';
      lightLogo.style.display = 'none';
      darkLogo.style.opacity = '1';
      darkLogo.style.display = 'block';
    } else {
      // For dark backgrounds, show light logo
      lightLogo.style.opacity = '1';
      lightLogo.style.display = 'block';
      darkLogo.style.opacity = '0';
      darkLogo.style.display = 'none';
    }
  }

  // Listen for slide changes
  document.addEventListener('ws:slide-change', updateLogoVisibility);
  
  // Poll for changes as a fallback (in case event isn't fired)
  let lastSlideId = '';
  setInterval(function() {
    const currentSlide = document.querySelector('#webslides section.current');
    if (currentSlide && currentSlide.id !== lastSlideId) {
      lastSlideId = currentSlide.id;
      updateLogoVisibility();
    }
  }, 500);

  // Handle initial state
  document.addEventListener('DOMContentLoaded', function() {
    // Try immediately
    updateLogoVisibility();
    
    // And also after a short delay to ensure WebSlides is fully initialized
    setTimeout(updateLogoVisibility, 500);
  });
  
  // Also update on window load as a final fallback
  window.addEventListener('load', updateLogoVisibility);
})();
