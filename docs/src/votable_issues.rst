.. include:: common_links.inc

.. _votable_issues:

Votable Issues
==============

You can help us to prioritise development of new features by leaving a 👍
reaction on the  header (not subsequent comments) of any issue that has a
label of ``Feature: Votable``.

Below is a list of all current votable enhancement issues from our github
project ordered by the amount of 👍.

Please note that there is more development activity than what is on the below
table, this is focusing only on the votable issues.

.. note:: The data in this table is updated daily and is sourced from
          `votable-issues.json`_.
          For the latest data please see the `votable issues on GitHub`_.
          Note that the list on Github does not show the number of votes 👍
          only the total number of comments for the whole issue.

.. _votable-issues.json: https://github.com/scitools/votable_issues/blob/main/votable-issues.json

.. raw:: html

   <table id="example" class="display" style="width:100%">
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
           $('#example').DataTable( {
              <!-- "ajax": 'votable-issues.json', -->
              "ajax": 'https://raw.githubusercontent.com/scitools/votable_issues/main/votable-issues.json',
              "pageLength": 20,
              "order": [[ 0, "desc" ]]
           } );
        } );
   </script>

