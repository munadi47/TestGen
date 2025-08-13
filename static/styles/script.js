document.addEventListener('DOMContentLoaded', function() {
    // Select all navigation list items in the sidebar
    const navItems = document.querySelectorAll('.nav-item');

    // Get the filename of the current page from the URL
    // e.g., for 'http://example.com/dashboard.html', currentPath will be 'dashboard.html'
    const currentPath = window.location.pathname.split('/').pop();

    // Iterate over each navigation item
    navItems.forEach(item => {
        // Find the anchor tag (link) within the current navigation item
        const link = item.querySelector('a');

        // Check if a link exists for the item
        if (link) {
            // Get the href attribute of the link
            const href = link.getAttribute('href');

            // Compare the link's href with the current page's filename
            // If they match, add the 'active' class to the navigation item
            const linkPath = href.startsWith('/') ? href.substring(1) : href;
            if (linkPath === currentPath) {
                item.classList.add('active');
            }

            // if (href === currentPath) {
            //     item.classList.add('active');
            // }

            item.addEventListener('click', function() {
                // Remove 'active' class from all other navigation items
                navItems.forEach(nav => nav.classList.remove('active'));
                // Add 'active' class to the clicked item
                this.classList.add('active');
            });
            
        }
    });

    
    fetch('/refinement.html', { /* ... options ... */ })
    .then(response => response.json())
    .then(data => {
        // 1. Check the raw data from the server
        console.log("Data received from server:", data);
        console.log("Markdown string:", data.refined_markdown);

        // 2. This is where the error likely is.
        // Check the function that parses the markdown into an array.
        const tableDataArray = parseMarkdownToJSArray(data.refined_markdown);

        // 3. Check the array BEFORE it goes to the table
        console.log("Parsed array before sending to table:", tableDataArray);
        console.log("Number of rows:", tableDataArray.length); // <-- This will probably show 2 instead of 21

        // 4. Initialize the table
        $('#myTable').DataTable({
            data: tableDataArray,
            // ... other options
        });
    });
});