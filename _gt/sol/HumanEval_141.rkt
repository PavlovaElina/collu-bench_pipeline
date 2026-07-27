#lang racket
(define (file_name_check file_name)
  (define digit-count (for/sum ([c (in-string file_name)] #:when (char-numeric? c)) 1))
  (define parts (string-split file_name "." #:trim? #f))
  (cond [(> digit-count 3) "No"]
        [(not (= (length parts) 2)) "No"]
        [else (define before (first parts))
              (define after (second parts))
              (if (and (> (string-length before) 0)
                       (char-alphabetic? (string-ref before 0))
                       (member after '("txt" "exe" "dll")))
                  "Yes" "No")]))
