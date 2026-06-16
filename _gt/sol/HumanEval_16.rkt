#lang racket
(define (count_distinct_characters s)
  (length (remove-duplicates (map char-downcase (string->list s)))))
