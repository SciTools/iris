.. include:: common_links.inc

.. _voted_issues_top:

Voted Issues
============

You can help us to prioritise development of new features by leaving a 👍
reaction on the header (not subsequent comments) of any issue.

.. tip:: We suggest you subscribe to the issue so you will be updated.
         When viewing the issue there is a **Notifications**
         section where you can select to subscribe.

Below is a sorted table of all issues that have 1 or more 👍 from our github
project.  Please note that there is more development activity than what is on
the below table.

.. _voted-issues.json: https://github.com/scitools/voted_issues/blob/main/voted-issues.json

.. raw:: html

   <table id="voted_issues_table" class="hover row-border order-column" style="width:100%">
      <thead>
         <tr>
            <th>👍</th>
            <th>Issue</th>
            <th>Author</th>
            <th>Title</th>
         </tr>
      </thead>
   </table>

   <!-- JS to enable the datatable features: sortable, paging, search etc
           https://datatables.net/reference/option/
           https://datatables.net/  -->

   <script type="text/javascript">
        $(document).ready(function() {
           $('#voted_issues_table').DataTable( {
              <!-- "ajax": 'voted-issues.json', -->
              "ajax": 'https://raw.githubusercontent.com/scitools/voted_issues/main/voted-issues.json',
              "lengthMenu": [10, 25, 50, 100],
              "pageLength": 10,
              "order": [[ 0, "desc" ]],
              "bJQueryUI": true,
           } );
        } );
   </script>
   <p></p>


.. note:: The data in this table is updated every 30 minutes and is sourced
          from `voted-issues.json`_.
          For the latest data please see the `issues on GitHub`_.
          Note that the list on Github does not show the number of votes 👍
          only the total number of comments for the whole issue.